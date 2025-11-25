struct Geo
{
    vec3 pos;
    vec3 scale;
    vec3 rotation;
};

struct Ray { vec3 o, d; };
struct Camera { vec3 p, t; };
struct Hit { vec3 p; float t, d; };

Geo _geo = Geo(vec3(0),vec3(1),vec3(0));
Ray _ray;
Camera _cam;

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    struct Foo { vec3 a; float b, c; };
    Foo _bar =  Foo( vec3(0), 1.0, 0.5);
    Hit _hit = Hit( vec3(0), 1.0, 0.5);
    _bar.a = _hit.p;
    float x = _hit.t;
    vec3 col = _cam.p;
    fragColor = vec4(col,1.0);
}