// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
float2 uv = fragCoord.xy / iResolution.xy;
   
    uv *=  1.0f - uv.yx;   //float2(1.0f)- uv.yx; -> 1.-u.yx; Thanks FabriceNeyret !
    
    float vig = uv.x*uv.y * 15.0f; // multiply with sth for intensity
    
    vig = GLSL_pow(vig, 0.25f); // change pow for modifying the extend of the  vignette

    
    fragColor = (float4)(vig);
// ---- SHADERTOY CODE END ----