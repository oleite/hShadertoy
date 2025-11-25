// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 uv = fragCoord / iResolution.y; // Normalize coordinates
    float gridSize = float(20.0f);           // Grid size for Voronoi cells
    float2 gv = gol_fract2(uv * gridSize);       // Grid cell frac
    float2 id = floor(uv * gridSize);       // Grid cell

    float minDist2 = 1.0f;
    float3 color = (float3)(0.0f);
    float2 closestPoint;

    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            float2 offset = (float2)(x, y);
            float2 neighbor = id + offset;

            // Hash values
            float h = hash(neighbor);
            float2 point = hash2(neighbor);

            float angle = iTime + h * 6.2831f;
            float2 sincos = sin(angle + (float2)(0.0f, 1.5708f));
            point += 0.5f * sincos;

            float2 diff = offset + point - gv;
            float dist2 = dot(diff, diff);

            if(dist2 < minDist2) {
                minDist2 = dist2;
                closestPoint = diff;
                color = hash3(neighbor);
            }
        }
    }

    // Calculate normal & lighting
    float3 normal = normalize((float3)(closestPoint, 0.5f));
    float3 lightDir = normalize((float3)(0.5f, 0.5f, 1.0f));
    float lighting = max(dot(normal, lightDir), 0.0f);

    // Tada
    fragColor = (float4)(color * lighting, 1.0f);
// ---- SHADERTOY CODE END ----