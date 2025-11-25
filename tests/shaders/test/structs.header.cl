typedef struct {
    float3 pos;
    float3 scale;
    float3 rotation;
} Geo;

typedef struct {
    float3 o, d;
} Ray;

typedef struct {
    float3 p, t;
} Camera;

typedef struct {
    float3 p;
    float t, d;
} Hit;

Geo _geo = {(float3)(0), (float3)(1), (float3)(0)};

Ray _ray;

Camera _cam;

