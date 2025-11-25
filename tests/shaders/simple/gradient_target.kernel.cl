// ---- SHADERTOY CODE BEGIN ----
// Shadertoy void mainImage(...)
//mouse controlled version
    //float d = dist(iResolution.xy*0.5f,fragCoord.xy)*(iMouse.x/iResolution.x+0.1f)*0.01f;
    
    //automatic version
    float d = dist(iResolution.xy*0.5f,fragCoord.xy)*(sin(iTime)+1.5f)*0.003f;
	fragColor = GLSL_mix((float4)(1.0f, 1.0f, 1.0f, 1.0f), (float4)(0.0f, 0.0f, 0.0f, 1.0f), d);
// ---- SHADERTOY CODE END ----