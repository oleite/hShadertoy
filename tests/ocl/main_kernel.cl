

// SHADERTOY COMMON TAB

// RENDERPASS
// ---- SHADERTOY HEADER functions and macros HERE
// Transpiled Shadertoy header functions and macros

kernel void generickernel( 
    int2 _bound_tilesize,
    float _bound_time,
    fpreal4 _bound_iDate,
    fpreal  _bound_iFrame,
    fpreal  _bound_iFrameRate,
    fpreal4 _bound_iMouse,
#ifdef HAS_size_ref
    global void * restrict _bound_size_ref_stat_void,
    global void * restrict _bound_size_ref,
#else
    float4 _bound_size_ref,
#endif
#ifdef HAS_fragCoord
    global void * restrict _bound_fragCoord_stat_void,
    global void * restrict _bound_fragCoord,
#else
    float2 _bound_fragCoord,
#endif
#ifdef HAS_iChannel0
    global void * restrict _bound_iChannel0_stat_void,
    global void * restrict _bound_iChannel0,
#else
    float4 _bound_iChannel0,
#endif
#ifdef HAS_iChannel1
    global void * restrict _bound_iChannel1_stat_void,
    global void * restrict _bound_iChannel1,
#else
    float4 _bound_iChannel1,
#endif
#ifdef HAS_iChannel2
    global void * restrict _bound_iChannel2_stat_void,
    global void * restrict _bound_iChannel2,
#else
    float4 _bound_iChannel2,
#endif
#ifdef HAS_iChannel3
    global void * restrict _bound_iChannel3_stat_void,
    global void * restrict _bound_iChannel3,
#else
    float4 _bound_iChannel3,
#endif
    global void * restrict _bound_fragColor_stat_void,
    global void * restrict _bound_fragColor
)
{
#ifdef HAS_size_ref
    IMX_Layer _bound_size_ref_layer = {_bound_size_ref, _bound_size_ref_stat_void};
#endif
#ifdef HAS_fragCoord
    IMX_Layer _bound_fragCoord_layer = {_bound_fragCoord, _bound_fragCoord_stat_void};
#endif
#ifdef HAS_iChannel0
    IMX_Layer _bound_iChannel0_layer = {_bound_iChannel0, _bound_iChannel0_stat_void};
#endif
#ifdef HAS_iChannel1
    IMX_Layer _bound_iChannel1_layer = {_bound_iChannel1, _bound_iChannel1_stat_void};
#endif
#ifdef HAS_iChannel2
    IMX_Layer _bound_iChannel2_layer = {_bound_iChannel2, _bound_iChannel2_stat_void};
#endif
#ifdef HAS_iChannel3
    IMX_Layer _bound_iChannel3_layer = {_bound_iChannel3, _bound_iChannel3_stat_void};
#endif
    IMX_Layer _bound_fragColor_layer = {_bound_fragColor, _bound_fragColor_stat_void};
    int _bound_gidx = get_global_id(0) * _bound_tilesize.x;
    int _bound_gidy = get_global_id(1) * _bound_tilesize.y;
    if (_bound_gidx >= _RUNOVER_LAYER.stat->resolution.x)
        return;
    if (_bound_gidy >= _RUNOVER_LAYER.stat->resolution.y)
        return;
    int _bound_idx = _linearIndex(_RUNOVER_LAYER.stat, (int2)(_bound_gidx, _bound_gidy));
    float2 _bound_P_image = bufferToImage(_RUNOVER_LAYER.stat, (float2)(_bound_gidx, _bound_gidy));
    float2 _bound_P_texture = bufferToTexture(_RUNOVER_LAYER.stat, (float2)(_bound_gidx, _bound_gidy));
    float2 _bound_P_pixel = bufferToPixel(_RUNOVER_LAYER.stat, (float2)(_bound_gidx, _bound_gidy));

#line 80
   
   SHADERTOY_INPUTS
    // Shadertoy inputs defined in main_header.cl 
    // const float3    iResolution;           // viewport resolution (in pixels)
    // const float     iTime;                 // shader playback time (in seconds)
    // const float     iTimeDelta;            // render time (in seconds)
    // const float     iFrameRate;            // shader frame rate
    // const int       iFrame;                // shader playback frame
    // const float     iChannelTime[4];       // channel playback time (in seconds)
    // const float3    iChannelResolution[4]; // channel resolution (in pixels)
    // const float4    iMouse;                // mouse pixel coords. xy: current (if MLB down), zw: click
    // const samplerXX iChannel0..3;          // input channel. XX = 2D/Cube
    // const float4    iDate;                 // (year, month, day, time in seconds)
    // const float     iSampleRate;           // sound sample rate (i.e., 44100)
    //
    // float4 fragColor
    // float2 fragCoord
    
    // ---- SHADERTOY CODE BEGIN ----
    
//  Kernel end footer appended in compilecl.py construct_kernel_source()
//  AT_fragColor_set(fragColor);}