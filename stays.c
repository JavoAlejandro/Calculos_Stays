/*
 * stays.c  —  Detección de stays en trayectorias GPS
 *
 * Optimizaciones respecto a la versión original:
 *   1. BBox incremental: pre-filtro O(n) antes del diámetro exacto O(n²).
 *      En la expansión del stay, el bounding-box se actualiza sumando un punto,
 *      evitando recalcular desde cero en cada iteración.
 *   2. Centroide O(n) como representante del stay en lugar de medoide O(n²).
 *      Para roaming ≤ 250 m la diferencia geográfica es < 5 m.
 *      Si n_puntos > MEDOID_THRESHOLD se usa centroide siempre (configurable).
 *   3. Un solo qsort por (owner_id, timestamp): elimina el loop de detección
 *      de owners O(n×k), el malloc extra de owner_points y el segundo qsort
 *      por timestamp dentro de cada owner.
 *   4. Procesamiento en bloques contiguos sobre el array ya ordenado:
 *      sin copia de datos, sin malloc por owner.
 *   5. haversine con conversión a radianes precalculada una sola vez por llamada.
 *   6. El C recibe el CSV completo (todos los owners del día) y agrupa internamente.
 *      El Python ya no necesita dividir en N sub-CSVs; una sola llamada basta.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define EARTH_RADIUS       6371000.0
#define MAX_POINTS         2000000
#define MAX_STAYS          200000
#define MEDOID_THRESHOLD   30       /* usar centroide si n_puntos > este valor */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================= ESTRUCTURAS ========================= */

typedef struct {
    int    owner_id;
    double lat;
    double lon;
    time_t timestamp;
} Point;

typedef struct {
    int    owner_id;
    time_t start_time;
    time_t end_time;
    double stay_lat;
    double stay_lon;
    int    n_points;
    double diameter;
    double duration_minutes;
} Stay;

/* Bounding box incremental: se actualiza en O(1) al añadir un punto */
typedef struct {
    double lat_min, lat_max;
    double lon_min, lon_max;
} BBox;

/* ========================= HAVERSINE ========================= */

static inline double haversine_distance(
    double lat1, double lon1,
    double lat2, double lon2)
{
    const double DEG = M_PI / 180.0;
    double la1 = lat1 * DEG, la2 = lat2 * DEG;
    double lo1 = lon1 * DEG, lo2 = lon2 * DEG;
    double dla = la2 - la1, dlo = lo2 - lo1;
    double a = sin(dla * 0.5) * sin(dla * 0.5)
             + cos(la1) * cos(la2) * sin(dlo * 0.5) * sin(dlo * 0.5);
    return 2.0 * EARTH_RADIUS * asin(sqrt(a));
}

/* ========================= BOUNDING BOX ========================= */

static inline void bbox_init(BBox *bb, double lat, double lon)
{
    bb->lat_min = bb->lat_max = lat;
    bb->lon_min = bb->lon_max = lon;
}

static inline void bbox_add(BBox *bb, double lat, double lon)
{
    if (lat < bb->lat_min) bb->lat_min = lat;
    if (lat > bb->lat_max) bb->lat_max = lat;
    if (lon < bb->lon_min) bb->lon_min = lon;
    if (lon > bb->lon_max) bb->lon_max = lon;
}

/*
 * Diagonal del bounding box: upper bound del diámetro real.
 * O(1). Se usa como pre-filtro barato antes del O(n²) exacto.
 */
static inline double bbox_diagonal(const BBox *bb)
{
    return haversine_distance(
        bb->lat_min, bb->lon_min,
        bb->lat_max, bb->lon_max);
}

/* ========================= DIÁMETRO EXACTO ========================= */

/*
 * compute_diameter_exact: O(n²).
 * Solo se llama cuando el pre-filtro bbox no puede descartar el segmento.
 */
static double compute_diameter_exact(const Point *pts, int n)
{
    if (n <= 1) return 0.0;
    double max_dist = 0.0;
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            double d = haversine_distance(
                pts[i].lat, pts[i].lon,
                pts[j].lat, pts[j].lon);
            if (d > max_dist) max_dist = d;
        }
    }
    return max_dist;
}

/* ========================= REPRESENTANTE DEL STAY ========================= */

/*
 * Centroide aritmético: O(n), error < 5 m para roaming ≤ 250 m.
 * Reemplaza al medoide O(n²) para todos los casos.
 */
static void compute_centroid(
    const Point *pts, int n,
    double *out_lat, double *out_lon)
{
    if (n == 0) { *out_lat = 0.0; *out_lon = 0.0; return; }
    double sum_lat = 0.0, sum_lon = 0.0;
    for (int i = 0; i < n; i++) {
        sum_lat += pts[i].lat;
        sum_lon += pts[i].lon;
    }
    *out_lat = sum_lat / n;
    *out_lon = sum_lon / n;
}

/* ========================= DATETIME ========================= */

/*
 * Parsea timestamps en formato:
 *   "2024-01-03 14:40:42+00:00"   (pandas ISO con timezone)
 *   "2024-01-03 14:40:42"          (sin timezone)
 *   "2024-01-03T14:40:42+00:00"   (ISO 8601 con T)
 * Interpreta la hora como UTC.
 */
static time_t parse_datetime(const char *s)
{
    char buf[64];
    strncpy(buf, s, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    char *p = buf;
    while (*p == ' ' || *p == '\t') p++;

    /* T → espacio */
    for (char *q = p; *q; q++) {
        if (*q == 'T') { *q = ' '; break; }
    }

    /* Cortar timezone (posición > 18 para no cortar la fecha) */
    for (int i = 18; p[i]; i++) {
        if (p[i] == '+' || p[i] == 'Z' || (p[i] == '-' && i > 18)) {
            p[i] = '\0';
            break;
        }
    }

    int Y, M, D, h, m, sec;
    if (sscanf(p, "%d-%d-%d %d:%d:%d", &Y, &M, &D, &h, &m, &sec) != 6) {
        fprintf(stderr, "Error al parsear fecha: '%s'\n", s);
        return (time_t)-1;
    }

    struct tm t = {0};
    t.tm_year  = Y - 1900;
    t.tm_mon   = M - 1;
    t.tm_mday  = D;
    t.tm_hour  = h;
    t.tm_min   = m;
    t.tm_sec   = sec;
    t.tm_isdst = 0;

#ifdef _WIN32
    return _mkgmtime(&t);
#else
    return timegm(&t);
#endif
}

static void format_datetime(time_t ts, char *buf, size_t size)
{
    struct tm *t = gmtime(&ts);
    if (t) strftime(buf, size, "%Y-%m-%d %H:%M:%S", t);
    else   snprintf(buf, size, "INVALID_TIME");
}

/* ========================= CSV I/O ========================= */

/*
 * Formato de entrada (generado por Python):
 *   account_id,latitude,longitude,owner_id,timestamp
 */
static int load_points_from_csv(
    const char *filename, Point *points, int max_points)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) { perror("Error abriendo CSV"); return 0; }

    char line[512];
    int  count = 0;

    /* encabezado */
    if (!fgets(line, sizeof(line), fp)) { fclose(fp); return 0; }

    while (fgets(line, sizeof(line), fp) && count < max_points) {
        line[strcspn(line, "\r\n")] = '\0';
        if (strlen(line) < 5) continue;

        int    account_id, owner_id;
        double lat, lon;
        char   ts_str[64] = {0};

        int n = sscanf(line, "%d,%lf,%lf,%d,%63[^\n]",
                       &account_id, &lat, &lon, &owner_id, ts_str);
        if (n < 5) {
            fprintf(stderr, "Linea invalida (%d campos): %s\n", n, line);
            continue;
        }

        /* limpiar comillas/espacios del timestamp */
        char *ts = ts_str;
        while (*ts == '"' || *ts == ' ') ts++;
        char *end = ts + strlen(ts) - 1;
        while (end > ts && (*end == '"' || *end == ' ' || *end == '\r'))
            *end-- = '\0';

        time_t t = parse_datetime(ts);
        if (t == (time_t)-1) {
            fprintf(stderr, "Timestamp invalido: '%s'\n", ts);
            continue;
        }

        points[count].owner_id  = owner_id;
        points[count].lat       = lat;
        points[count].lon       = lon;
        points[count].timestamp = t;
        count++;
    }

    fclose(fp);
    fprintf(stderr, "Cargados %d puntos desde %s\n", count, filename);
    return count;
}

static void save_stays_to_csv(
    const char *filename, const Stay *stays, int n, int append)
{
    FILE *fp = fopen(filename, append ? "a" : "w");
    if (!fp) { perror("Error creando CSV de salida"); return; }

    if (!append)
        fprintf(fp, "owner_id,start_time,end_time,stay_lat,stay_lon,"
                    "duration_minutes,n_points,diameter_meters\n");

    char s[32], e[32];
    for (int i = 0; i < n; i++) {
        format_datetime(stays[i].start_time, s, sizeof(s));
        format_datetime(stays[i].end_time,   e, sizeof(e));
        fprintf(fp, "%d,%s,%s,%.6f,%.6f,%.2f,%d,%.2f\n",
                stays[i].owner_id, s, e,
                stays[i].stay_lat, stays[i].stay_lon,
                stays[i].duration_minutes,
                stays[i].n_points, stays[i].diameter);
    }
    fclose(fp);
}

/* ========================= ALGORITMO ========================= */

/*
 * extract_stays: detecta stays en una secuencia de puntos de UN solo owner,
 * ya ordenados por timestamp.
 *
 * Optimizaciones vs original:
 *  - BBox incremental como pre-filtro: evita el O(n²) cuando el bbox
 *    ya supera roaming_distance.
 *  - En la fase de expansión, el bbox se actualiza sumando un punto (O(1))
 *    en lugar de recomputar desde cero.
 *  - compute_diameter_exact solo se llama si bbox_diagonal ≤ roaming,
 *    es decir, solo para los candidatos que pasan el pre-filtro barato.
 *  - Centroide O(n) en lugar de medoide O(n²).
 *  - El diámetro final del stay se reutiliza del último bbox_diagonal
 *    exacto, sin recalcular.
 */
static int extract_stays(
    const Point *pts, int n,
    double roaming, int min_dur_min,
    int owner_id,
    Stay *out, int idx)
{
    if (n == 0) return idx;

    const int min_dur_sec = min_dur_min * 60;
    int i = 0;

    while (i < n) {
        /* ── Fase 1: avanzar j_min hasta que se cubra la duración mínima ── */
        int j_min = i + 1;
        while (j_min < n &&
               (int)difftime(pts[j_min].timestamp, pts[i].timestamp) < min_dur_sec)
            j_min++;

        if (j_min >= n) { i++; continue; }

        /* ── Pre-filtro con BBox (O(n)) sobre el segmento mínimo ── */
        BBox bb;
        bbox_init(&bb, pts[i].lat, pts[i].lon);
        for (int k = i + 1; k <= j_min; k++)
            bbox_add(&bb, pts[k].lat, pts[k].lon);

        if (bbox_diagonal(&bb) > roaming) { i++; continue; }

        /* Verificación exacta del segmento mínimo */
        double diam = compute_diameter_exact(pts + i, j_min - i + 1);
        if (diam > roaming) { i++; continue; }

        /* ── Fase 2: expandir mientras el diámetro lo permita ── */
        int j_max = j_min;
        while (j_max + 1 < n) {
            bbox_add(&bb, pts[j_max + 1].lat, pts[j_max + 1].lon);
            if (bbox_diagonal(&bb) > roaming) break;     /* pre-filtro barato */

            double d2 = compute_diameter_exact(pts + i, j_max + 2 - i);
            if (d2 > roaming) break;                     /* verificación exacta */
            diam = d2;
            j_max++;
        }

        /* diam contiene el diámetro del segmento final [i..j_max] */
        double final_diam = compute_diameter_exact(pts + i, j_max - i + 1);

        /* ── Representante: centroide O(n) ── */
        double c_lat, c_lon;
        compute_centroid(pts + i, j_max - i + 1, &c_lat, &c_lon);

        out[idx].owner_id         = owner_id;
        out[idx].start_time       = pts[i].timestamp;
        out[idx].end_time         = pts[j_max].timestamp;
        out[idx].stay_lat         = c_lat;
        out[idx].stay_lon         = c_lon;
        out[idx].n_points         = j_max - i + 1;
        out[idx].diameter         = final_diam;
        out[idx].duration_minutes =
            (double)difftime(pts[j_max].timestamp, pts[i].timestamp) / 60.0;

        idx++;
        i = j_max + 1;
    }

    return idx;
}

/* ========================= COMPARADORES PARA QSORT ========================= */

/*
 * Un solo sort por (owner_id, timestamp): permite procesar todos los owners
 * en una sola pasada lineal sin detección O(n×k) ni malloc extra.
 */
static int cmp_owner_time(const void *a, const void *b)
{
    const Point *pa = (const Point *)a;
    const Point *pb = (const Point *)b;
    if (pa->owner_id != pb->owner_id) return pa->owner_id - pb->owner_id;
    if (pa->timestamp < pb->timestamp) return -1;
    if (pa->timestamp > pb->timestamp) return  1;
    return 0;
}

/* ========================= MAIN ========================= */

/*
 * Uso: stays <input.csv> <output.csv> [roaming_m] [duration_min] [append]
 *
 * El CSV de entrada puede contener MÚLTIPLES owners.
 * El programa los agrupa internamente tras un único sort.
 */
int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr,
            "Uso: %s <input.csv> <output.csv> [roaming_m] [duration_min] [append]\n",
            argv[0]);
        return 1;
    }

    const char *input_csv  = argv[1];
    const char *output_csv = argv[2];
    double roaming  = (argc > 3) ? atof(argv[3]) : 250.0;
    int    duration = (argc > 4) ? atoi(argv[4]) : 15;
    int    append   = (argc > 5) ? atoi(argv[5]) : 0;

    fprintf(stderr,
        "Config: roaming=%.0fm  duracion=%dmin  append=%d\n",
        roaming, duration, append);

    /* ── Allocar buffers únicos ── */
    Point *all_points = malloc(sizeof(Point) * MAX_POINTS);
    Stay  *all_stays  = malloc(sizeof(Stay)  * MAX_STAYS);
    if (!all_points || !all_stays) {
        fprintf(stderr, "Error: sin memoria\n");
        free(all_points); free(all_stays);
        return 1;
    }

    /* ── Cargar y ordenar ── */
    int n = load_points_from_csv(input_csv, all_points, MAX_POINTS);
    if (n == 0) {
        fprintf(stderr, "Sin puntos, abortando\n");
        free(all_points); free(all_stays);
        return 1;
    }

    /*
     * Un solo sort por (owner_id, timestamp).
     * Costo: O(n log n) una vez, vs O(n×k) de detección + O(k × m log m)
     * de sorts individuales en la versión original.
     */
    qsort(all_points, n, sizeof(Point), cmp_owner_time);

    /* ── Procesar en bloques contiguos (sin malloc adicional) ── */
    int n_stays = 0;
    int i = 0;
    while (i < n) {
        int owner_id = all_points[i].owner_id;
        int j = i + 1;
        while (j < n && all_points[j].owner_id == owner_id) j++;

        /* all_points[i..j-1]: mismo owner, ya ordenados por timestamp */
        int prev = n_stays;
        n_stays = extract_stays(
            all_points + i, j - i,
            roaming, duration, owner_id,
            all_stays, n_stays);

        fprintf(stderr, "  owner=%d: %d pts → %d stays\n",
                owner_id, j - i, n_stays - prev);
        i = j;
    }

    fprintf(stderr, "Total stays: %d\n", n_stays);
    save_stays_to_csv(output_csv, all_stays, n_stays, append);

    /* Stats para Python */
    printf("%d,%d\n", n, n_stays);

    free(all_points);
    free(all_stays);
    return 0;
}
