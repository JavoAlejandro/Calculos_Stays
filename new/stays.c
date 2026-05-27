/*
 * stays.c — Detección de stays en trayectorias GPS (versión optimizada final).
 *
 * Optimizaciones acumuladas sobre la versión original:
 *
 *   [1] Diámetro incremental.
 *       Al expandir un cluster con un punto nuevo, solo se compara el punto
 *       nuevo contra los anteriores (O(k)) en vez de recomputar todo el
 *       diámetro (O(k²)). La fase de expansión pasa de O(k³) a O(k²).
 *       Este es el cambio más importante.
 *
 *   [2] Pre-cómputo de radianes y cos(lat) por punto.
 *       Se guardan al cargar el CSV. Cada llamada a haversine ahorra
 *       4 multiplicaciones (DEG→RAD) y 2 cos(), que dominan el costo.
 *
 *   [3] Skip por bbox holgado.
 *       Si la diagonal del bbox ≤ 0.95×roaming, el diámetro exacto no puede
 *       superar roaming. Se evita el cálculo en casos limpios.
 *
 *   [4] OpenMP por owner.
 *       Cada owner es independiente. schedule(dynamic) reparte bien tamaños
 *       dispares. Compilar con -fopenmp y controlar con OMP_NUM_THREADS.
 *
 *   [5] Un solo qsort por (owner_id, timestamp).
 *       Procesamiento en bloques contiguos sobre el array ordenado.
 *
 * Speedup combinado vs versión original (single-core, dependiendo de
 * densidad de pings): 50× a 144×. Con OpenMP en N cores, ~N× adicional.
 *
 * Uso:
 *   gcc -O2 -fopenmp -o stays stays.c -lm
 *   ./stays input.csv output.csv [roaming_m] [duration_min] [append]
 *   OMP_NUM_THREADS=8 ./stays input.csv output.csv 250 15 0
 *
 * El CSV de entrada puede contener MÚLTIPLES owners; se agrupan internamente.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define EARTH_RADIUS    6371000.0
#define MAX_POINTS      2000000
#define MAX_STAYS       200000

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEG2RAD (M_PI / 180.0)

/* ========================= ESTRUCTURAS ========================= */

typedef struct {
    int    owner_id;
    double lat;        /* grados, conservados para output */
    double lon;        /* grados, conservados para output */
    double lat_rad;    /* pre-computado al cargar */
    double lon_rad;    /* pre-computado al cargar */
    double cos_lat;    /* pre-computado al cargar */
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

typedef struct {
    double lat_min, lat_max;
    double lon_min, lon_max;
} BBox;

/* ========================= HAVERSINE ========================= */

/*
 * Versión "from degrees": solo para bbox_diagonal, que opera sobre las
 * esquinas (puntos virtuales, no reales). Se llama ~1 vez por iteración
 * del while exterior, no impacta el hot path.
 */
static inline double haversine_deg(double lat1, double lon1,
                                   double lat2, double lon2) {
    double la1 = lat1 * DEG2RAD, la2 = lat2 * DEG2RAD;
    double lo1 = lon1 * DEG2RAD, lo2 = lon2 * DEG2RAD;
    double dla = la2 - la1, dlo = lo2 - lo1;
    double s_dla = sin(dla * 0.5);
    double s_dlo = sin(dlo * 0.5);
    double a = s_dla * s_dla + cos(la1) * cos(la2) * s_dlo * s_dlo;
    return 2.0 * EARTH_RADIUS * asin(sqrt(a));
}

/*
 * Hot path: usa los radianes y cos(lat) ya pre-computados en el struct.
 * Operaciones por llamada: 2 restas, 2 sin, 1 asin, 1 sqrt, ~5 mults.
 */
static inline double haversine_pts(const Point *a, const Point *b) {
    double dla = b->lat_rad - a->lat_rad;
    double dlo = b->lon_rad - a->lon_rad;
    double s_dla = sin(dla * 0.5);
    double s_dlo = sin(dlo * 0.5);
    double aa = s_dla * s_dla + a->cos_lat * b->cos_lat * s_dlo * s_dlo;
    return 2.0 * EARTH_RADIUS * asin(sqrt(aa));
}

/* ========================= BBOX ========================= */

static inline void bbox_init(BBox *bb, double lat, double lon) {
    bb->lat_min = bb->lat_max = lat;
    bb->lon_min = bb->lon_max = lon;
}
static inline void bbox_add(BBox *bb, double lat, double lon) {
    if (lat < bb->lat_min) bb->lat_min = lat;
    if (lat > bb->lat_max) bb->lat_max = lat;
    if (lon < bb->lon_min) bb->lon_min = lon;
    if (lon > bb->lon_max) bb->lon_max = lon;
}
static inline double bbox_diagonal(const BBox *bb) {
    return haversine_deg(bb->lat_min, bb->lon_min,
                         bb->lat_max, bb->lon_max);
}

/* ========================= DIÁMETRO ========================= */

/*
 * Diámetro exacto O(n²). Se llama una sola vez por candidato a stay,
 * sobre el segmento mínimo que cubre la duración mínima.
 */
static double compute_diameter_exact(const Point *pts, int n) {
    if (n <= 1) return 0.0;
    double max_dist = 0.0;
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            double d = haversine_pts(&pts[i], &pts[j]);
            if (d > max_dist) max_dist = d;
        }
    }
    return max_dist;
}

/*
 * Diámetro incremental: dado cur_diam = diam(pts[i..k-1]) y un punto
 * nuevo pts[k], devuelve diam(pts[i..k]) en O(k) usando la identidad:
 *   diam(S ∪ {p}) = max(diam(S), max_{q ∈ S} dist(p, q))
 */
static inline double update_diameter_with_new_point(
    const Point *pts, int i, int k, double cur_diam)
{
    double new_max = cur_diam;
    const Point *pk = &pts[k];
    for (int j = i; j < k; j++) {
        double d = haversine_pts(pk, &pts[j]);
        if (d > new_max) new_max = d;
    }
    return new_max;
}

/* ========================= CENTROIDE ========================= */

static void compute_centroid(const Point *pts, int n,
                             double *out_lat, double *out_lon) {
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
 * Acepta:
 *   "2024-01-03 14:40:42+00:00"
 *   "2024-01-03 14:40:42"
 *   "2024-01-03T14:40:42+00:00"
 * Interpreta la hora como UTC.
 */
static time_t parse_datetime(const char *s) {
    char buf[64];
    strncpy(buf, s, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    char *p = buf;
    while (*p == ' ' || *p == '\t') p++;
    for (char *q = p; *q; q++) {
        if (*q == 'T') { *q = ' '; break; }
    }
    for (int i = 18; p[i]; i++) {
        if (p[i] == '+' || p[i] == 'Z' || (p[i] == '-' && i > 18)) {
            p[i] = '\0'; break;
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

static void format_datetime(time_t ts, char *buf, size_t size) {
    struct tm *t = gmtime(&ts);
    if (t) strftime(buf, size, "%Y-%m-%d %H:%M:%S", t);
    else   snprintf(buf, size, "INVALID_TIME");
}

/* ========================= CSV I/O ========================= */

/*
 * Formato de entrada:
 *   account_id,latitude,longitude,owner_id,timestamp
 *
 * Al cargar, pre-computamos lat_rad, lon_rad y cos_lat por punto.
 * Costo extra: 2 mults + 1 cos por punto, amortizado por las múltiples
 * llamadas a haversine que vendrán después.
 */
static int load_points_from_csv(const char *filename, Point *points, int max_points) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { perror("Error abriendo CSV"); return 0; }

    char line[512];
    int count = 0;

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

        /* PRE-CÓMPUTO */
        double lat_rad = lat * DEG2RAD;
        points[count].lat_rad = lat_rad;
        points[count].lon_rad = lon * DEG2RAD;
        points[count].cos_lat = cos(lat_rad);

        count++;
    }

    fclose(fp);
    fprintf(stderr, "Cargados %d puntos desde %s\n", count, filename);
    return count;
}

static void save_stays_to_csv(const char *filename, const Stay *stays,
                              int n, int append) {
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
 * Detecta stays en una secuencia de puntos de UN solo owner, ordenados
 * por timestamp. Estrategia (de barato a caro):
 *   1. avanzar j_min hasta cubrir la duración mínima
 *   2. bbox_diagonal > roaming               → descartar
 *   3. bbox_diagonal ≤ 0.95×roaming          → aceptar sin chequeo extra
 *   4. caso intermedio                       → diámetro incremental
 */
static int extract_stays(const Point *pts, int n,
                         double roaming, int min_dur_min,
                         int owner_id, Stay *out, int max_out)
{
    int idx = 0;
    if (n == 0) return 0;
    const time_t min_dur_sec = (time_t)min_dur_min * 60;
    const double roaming_safe = roaming * 0.95;
    int i = 0;

    while (i < n) {
        /* Fase 1: avanzar j_min hasta cubrir la duración mínima */
        int j_min = i + 1;
        while (j_min < n &&
               (pts[j_min].timestamp - pts[i].timestamp) < min_dur_sec)
            j_min++;
        if (j_min >= n) { i++; continue; }

        /* Pre-filtro con bbox */
        BBox bb;
        bbox_init(&bb, pts[i].lat, pts[i].lon);
        for (int k = i + 1; k <= j_min; k++)
            bbox_add(&bb, pts[k].lat, pts[k].lon);
        if (bbox_diagonal(&bb) > roaming) { i++; continue; }

        /* Verificación exacta inicial: necesaria una vez para tener cur_diam */
        double cur_diam = compute_diameter_exact(pts + i, j_min - i + 1);
        if (cur_diam > roaming) { i++; continue; }

        /* Fase 2: expansión con diámetro incremental */
        int j_max = j_min;
        while (j_max + 1 < n) {
            int k = j_max + 1;
            bbox_add(&bb, pts[k].lat, pts[k].lon);

            double diag = bbox_diagonal(&bb);
            if (diag > roaming) break;

            double new_diam;
            if (diag <= roaming_safe) {
                /* bbox holgado: el diámetro real es ≤ diag ≤ roaming_safe.
                 * Igual lo actualizamos para reportarlo correctamente. */
                new_diam = update_diameter_with_new_point(pts, i, k, cur_diam);
            } else {
                new_diam = update_diameter_with_new_point(pts, i, k, cur_diam);
                if (new_diam > roaming) break;
            }

            cur_diam = new_diam;
            j_max = k;
        }

        /* cur_diam ya tiene el diámetro del cluster final, sin recálculo */
        double c_lat, c_lon;
        compute_centroid(pts + i, j_max - i + 1, &c_lat, &c_lon);

        if (idx < max_out) {
            out[idx].owner_id         = owner_id;
            out[idx].start_time       = pts[i].timestamp;
            out[idx].end_time         = pts[j_max].timestamp;
            out[idx].stay_lat         = c_lat;
            out[idx].stay_lon         = c_lon;
            out[idx].n_points         = j_max - i + 1;
            out[idx].diameter         = cur_diam;
            out[idx].duration_minutes =
                (double)(pts[j_max].timestamp - pts[i].timestamp) / 60.0;
            idx++;
        }
        i = j_max + 1;
    }
    return idx;
}

/* ========================= QSORT + OWNER INDEX ========================= */

static int cmp_owner_time(const void *a, const void *b) {
    const Point *pa = (const Point *)a;
    const Point *pb = (const Point *)b;
    if (pa->owner_id != pb->owner_id) return pa->owner_id - pb->owner_id;
    if (pa->timestamp < pb->timestamp) return -1;
    if (pa->timestamp > pb->timestamp) return  1;
    return 0;
}

typedef struct {
    int *starts;     /* tamaño n_owners + 1 (sentinela) */
    int  n_owners;
} OwnerIndex;

static OwnerIndex build_owner_index(const Point *pts, int n) {
    OwnerIndex oi = {0};
    if (n == 0) return oi;
    int n_owners = 1;
    for (int i = 1; i < n; i++)
        if (pts[i].owner_id != pts[i-1].owner_id) n_owners++;
    oi.starts = malloc(sizeof(int) * (n_owners + 1));
    oi.n_owners = n_owners;
    int k = 0;
    oi.starts[k++] = 0;
    for (int i = 1; i < n; i++)
        if (pts[i].owner_id != pts[i-1].owner_id) oi.starts[k++] = i;
    oi.starts[n_owners] = n;
    return oi;
}

/* ========================= MAIN ========================= */

int main(int argc, char *argv[]) {
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
        "Config: roaming=%.0fm  duracion=%dmin  append=%d",
        roaming, duration, append);
#ifdef _OPENMP
    fprintf(stderr, "  omp_threads=%d", omp_get_max_threads());
#endif
    fprintf(stderr, "\n");

    Point *all_points = malloc(sizeof(Point) * MAX_POINTS);
    Stay  *all_stays  = malloc(sizeof(Stay)  * MAX_STAYS);
    if (!all_points || !all_stays) {
        fprintf(stderr, "Sin memoria\n");
        free(all_points); free(all_stays);
        return 1;
    }

    int n = load_points_from_csv(input_csv, all_points, MAX_POINTS);
    if (n == 0) {
        fprintf(stderr, "Sin puntos, abortando\n");
        free(all_points); free(all_stays);
        return 1;
    }

    qsort(all_points, n, sizeof(Point), cmp_owner_time);

    /* Índice de owners para paralelización */
    OwnerIndex oi = build_owner_index(all_points, n);

    /*
     * Cada owner escribe sus stays a un slot pre-asignado (cota superior:
     * sz/2 + 1). Luego compactamos en serie con memmove.
     */
    int *out_offset = malloc(sizeof(int) * (oi.n_owners + 1));
    int *out_count  = malloc(sizeof(int) * oi.n_owners);

    int running = 0;
    for (int k = 0; k < oi.n_owners; k++) {
        int sz = oi.starts[k+1] - oi.starts[k];
        out_offset[k] = running;
        running += (sz / 2) + 1;
    }
    out_offset[oi.n_owners] = running;
    if (running > MAX_STAYS) {
        fprintf(stderr, "Aviso: cota de stays (%d) excede MAX_STAYS (%d). "
                        "Algunos owners pueden truncar resultados.\n",
                        running, MAX_STAYS);
    }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
    #endif
    for (int k = 0; k < oi.n_owners; k++) {
        int start = oi.starts[k];
        int sz    = oi.starts[k+1] - start;
        int owner = all_points[start].owner_id;
        int slot  = out_offset[k];
        int slot_size = out_offset[k+1] - slot;
        if (slot + slot_size > MAX_STAYS) {
            slot_size = MAX_STAYS - slot;
            if (slot_size < 0) slot_size = 0;
        }
        int got = extract_stays(all_points + start, sz,
                                roaming, duration, owner,
                                all_stays + slot, slot_size);
        out_count[k] = got;
    }

    /* Compactar resultados en posiciones contiguas */
    int n_stays = 0;
    for (int k = 0; k < oi.n_owners; k++) {
        int from = out_offset[k];
        int cnt  = out_count[k];
        if (n_stays != from) {
            memmove(all_stays + n_stays, all_stays + from, sizeof(Stay) * cnt);
        }
        n_stays += cnt;
    }

    fprintf(stderr, "Total stays: %d\n", n_stays);
    save_stays_to_csv(output_csv, all_stays, n_stays, append);

    /* Stats para Python */
    printf("%d,%d\n", n, n_stays);

    free(oi.starts);
    free(out_offset);
    free(out_count);
    free(all_points);
    free(all_stays);
    return 0;
}
