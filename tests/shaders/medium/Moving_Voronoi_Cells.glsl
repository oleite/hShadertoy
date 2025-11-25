// Moving Voronoi Cells
// https://www.shadertoy.com/view/lccyD8
// Hash Functions
float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec2 hash2(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract(vec2((p3.x + p3.y) * p3.z, (p3.y + p3.z) * p3.x));
}

vec3 hash3(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx + vec3(1.0, 2.0, 3.0)) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3 + p3.yzx) * p3.zxy);
}

// Main
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.y; // Normalize coordinates
    float gridSize = float(20.0);           // Grid size for Voronoi cells
    vec2 gv = fract(uv * gridSize);       // Grid cell frac
    vec2 id = floor(uv * gridSize);       // Grid cell

    float minDist2 = 1.0;
    vec3 color = vec3(0.0);
    vec2 closestPoint;

    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            vec2 offset = vec2(x, y);
            vec2 neighbor = id + offset;

            // Hash values
            float h = hash(neighbor);
            vec2 point = hash2(neighbor);

            float angle = iTime + h * 6.2831;
            vec2 sincos = sin(angle + vec2(0.0, 1.5708));
            point += 0.5 * sincos;

            vec2 diff = offset + point - gv;
            float dist2 = dot(diff, diff);

            if(dist2 < minDist2) {
                minDist2 = dist2;
                closestPoint = diff;
                color = hash3(neighbor);
            }
        }
    }

    // Calculate normal & lighting
    vec3 normal = normalize(vec3(closestPoint, 0.5));
    vec3 lightDir = normalize(vec3(0.5, 0.5, 1.0));
    float lighting = max(dot(normal, lightDir), 0.0);

    // Tada
    fragColor = vec4(color * lighting, 1.0);
}