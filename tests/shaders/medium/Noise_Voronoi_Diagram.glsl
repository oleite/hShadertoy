// Noise - Voronoi Diagram
// https://www.shadertoy.com/view/wcdXRf
// Gradient Generation
// Generate gradient vecors using a permutation polynomial
float permute(float x, float b) {
    float h = mod(x, 289.0f);
    return mod((34.0 * h + b) * h, 289.0);
}
vec3 gradient(vec3 p) {
    float B1 = 11.f;
    float B2 = 134.f;
    float B3 = 53.f;
    
    float x = permute(permute(permute(p.x, B1) + p.y, B1) + p.z, B1) / 288.f;
    float y = permute(permute(permute(p.x, B2) + p.y, B2) + p.z, B2) / 288.f;
    float z = permute(permute(permute(p.x, B3) + p.y, B3) + p.z, B3) / 288.f;
    
    // Our permutation function gives us a value in [0,1].
    // Convert this to a gradient vector that can go in any direction. 
    return normalize(vec3(x,y,z) - vec3(0.5, 0.5, 0.5));
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;
    vec3 xyz = vec3(uv * 15., iTime);
    
    // Config
    float jitter = 0.55f;
    
    // Determine the cell we are in
    vec3 cur_cell = floor(xyz);
    float c = 0.;
    
    float d1 = 500.;
    
    // Iterate through the neighbors of the cell
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            for (int k = -1; k <= 1; k++) {
                vec3 cell = cur_cell + vec3(i,j,k);

                // Compute the pseudorandom point from that cell
                vec3 jitter_vec = gradient(cell) * jitter;
                vec3 point = cell + vec3(0.5) + jitter_vec;

                // Compute distance to this point
                float d = length(point - xyz);

                // Update d1, d2. D1 is distance to closest, D2
                // is 2nd closest
                d1 = (d < d1) ? d : d1;
            }
        }
    }
    
    c = 1. - 2. * d1 * d1;
    
    
    // Output to screen
    vec3 color = mix(vec3(0.1), vec3(0.8, 0.1, 0.2), c);
    fragColor = vec4(color, 1.);
}