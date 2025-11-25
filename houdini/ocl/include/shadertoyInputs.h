// Shadertoy has global variables that can be called inside functions
// We just initiate empty variables so that code compiles if used inside func()
// They get mapped inside kernel
static float3 iResolution = (float3)(512.0f, 288.0f, 0.0f);
static float iTime = 0.0000f;
static float iTimeDelta = 0.0000f;
static float iFrameRate = 24.0000f;
static int iFrame = 0;
static float4 iMouse = (float4)(0.0000f, 0.0000f, 0.0000f, 0.0000f );
static float4 iDate = (float4)(2025.0000f, 12.0000f, 31.0000f, 60.0000f );
static const float iSampleRate = 44100.0f;

static const IMX_Layer* iChannel0;
static const IMX_Layer* iChannel1;
static const IMX_Layer* iChannel2;
static const IMX_Layer* iChannel3;
static float iChannelTime[4];
static float3 iChannelResolution[4];


#ifdef CUBEMAP_RENDERPASS
    #define DO_CUBEMAP \
        float3 rayDir; \
        shadertoy_cubemap(AT_ix,AT_iy,AT_xres,AT_yres,&rayDir,&iResolution);
#else
    #define DO_CUBEMAP /* nothing */
#endif

#define SHADERTOY_INPUTS \
    iResolution = (float3)(AT_xres, AT_yres, 0.0f); \
    iTime = AT_Time; \
    iFrameRate = AT_iFrameRate; \
    iFrame = AT_iFrame; \
    iMouse = AT_iMouse;\
    iDate = AT_iDate;\
    iChannel0 = AT_iChannel0_layer; \
    iChannel1 = AT_iChannel1_layer; \
    iChannel2 = AT_iChannel2_layer; \
    iChannel3 = AT_iChannel3_layer; \
    iChannelTime[0] = AT_Time; \
    iChannelTime[1] = AT_Time; \
    iChannelTime[2] = AT_Time; \
    iChannelTime[3] = AT_Time; \
    iChannelResolution[0] = (float3)(AT_iChannel0_res, 0.0f); \
    iChannelResolution[1] = (float3)(AT_iChannel1_res, 0.0f); \
    iChannelResolution[2] = (float3)(AT_iChannel2_res, 0.0f); \
    iChannelResolution[3] = (float3)(AT_iChannel3_res, 0.0f); \
    float2 fragCoord = AT_fragCoord; \
    if (!AT_fragCoord_bound) { fragCoord = (float2)(AT_ix, AT_iy); }\
    float4 fragColor = (float4)(0.0f, 0.0f, 0.0f, 1.0f); \
    DO_CUBEMAP

// mainCubemap renderpass helper    
// Unpacks 3x2 cubemap layout to ray direction and adjusts resolution to cube face
// Standard cubemap layout:
//   [+X][-X][+Z]
//   [+Y][-Y][-Z]
static void shadertoy_cubemap(int ix, int iy, int xres, int yres, 
                              float3* rayDir, float3* iResolution)
{
    // Calculate individual face dimensions
    int face_width = xres / 3;
    int face_height = yres / 2;
    
    // Update iResolution to single face size
    *iResolution = (float3)(face_width, face_height, 0.0f);
    
    // Determine which face we're rendering (0-2 for x, 0-1 for y)
    int face_x = ix / face_width;
    int face_y = iy / face_height;
    
    // Calculate local UV coordinates within the face (-1 to 1 range)
    float2 local_uv = (float2)(
        (float)(ix % face_width) / (float)face_width * 2.0f - 1.0f,
        (float)(iy % face_height) / (float)face_height * 2.0f - 1.0f
    );
    
    // Map face position to ray direction
    // Each face represents a direction in the cube
    if (face_x == 0 && face_y == 0) {
        // +X face (right)
        *rayDir = (float3)(1.0f, -local_uv.y, -local_uv.x);
    } 
    else if (face_x == 1 && face_y == 0) {
        // -X face (left)
        *rayDir = (float3)(-1.0f, -local_uv.y, local_uv.x);
    } 
    else if (face_x == 0 && face_y == 1) {
        // +Y face (up)
        *rayDir = (float3)(local_uv.x, 1.0f, local_uv.y);
    } 
    else if (face_x == 1 && face_y == 1) {
        // -Y face (down)
        *rayDir = (float3)(local_uv.x, -1.0f, -local_uv.y);
    }
    else if (face_x == 2 && face_y == 0) {
        // +Z face (forward)
        *rayDir = (float3)(local_uv.x, -local_uv.y, 1.0f);
    } 
    else if (face_x == 2 && face_y == 1) {
        // -Z face (back)
        *rayDir = (float3)(-local_uv.x, -local_uv.y, -1.0f);
    }
}