// 3D Voronoi noise 
// https://www.shadertoy.com/view/43dcWf
// Scroll to bottom for mainImage() function

const float ROOT_3 = sqrt(3.0);

// Calculate the coord of a hex cell in scaled and skewed grid space
vec3 hexCoord(vec3 p, float hexSize) {
    // Scale the input space by the hexagon size
    vec3 scaledP = p / hexSize;

    // Calculate approximate hex grid coordinates (using skewing)
    float x = ((ROOT_3 * scaledP.x) - scaledP.y - (scaledP.z * 2.0 / 3.0)) / 3.0;
    float y = (2.0 * (scaledP.y - (scaledP.z / 3.0))) / 3.0;
    float z = (2.0 * scaledP.z) / 3.0;

    // Round to the nearest hexagonal grid point
    float rx = round(x);
    float ry = round(y);
    float rz = round(z);

    float xDiff = rx - x;
    float yDiff = ry - y;
    float zDiff = rz - z;

    // 2D: skewed square -> hexagon prism
    float xShift = abs(xDiff * 2.0 + yDiff + zDiff);
    float yShift = abs(xDiff + yDiff * 2.0 + zDiff);
    if (xShift > 1.0 || yShift > 1.0) {
        if (xShift > yShift) {
            rx -= sign(xDiff);
        } else {
            ry -= sign(yDiff);
        }
    }

    return vec3(rx, ry, rz);
}

// Offset the coord by adding a hash centered around 0.0
vec3 offsetCoord(vec3 coord, float agitation) {
    vec3 d = vec3(
        dot(coord, vec3(123.1, 311.7, 741.7)),
        dot(coord, vec3(269.7, 183.3, 317.9)),
        dot(coord, vec3(147.3, 292.1, 457.1))
    );
    vec3 f = fract(sin(d) * 43758.5453) - 0.5;
    return coord + f * agitation;
}



// Unskew and unscale a hex coord
vec3 unskew(vec3 hexCoord, float hexSize) {
    vec3 scaledHexCenter = vec3(
        (hexCoord.x + ((hexCoord.y + hexCoord.z) * 0.5)) * ROOT_3,
        (hexCoord.y * 1.5) + (hexCoord.z * 0.5),
        hexCoord.z * 1.5
    );
    return scaledHexCenter * hexSize;
}



float hexDistance(vec3 p, vec3 center, float hexSize) {
    return length(p - center) / hexSize;
}

vec2 minDistances(vec3 p, vec3 hexCoord, vec3 neighborOffset, vec2 minDists, float hexSize, float agitation) {
    vec3 neighborCoord = hexCoord + neighborOffset;
    vec3 offsetNeighbor = offsetCoord(neighborCoord, agitation);
    vec3 unskewedNeighbor = unskew(offsetNeighbor, hexSize);
    float dist = hexDistance(p, unskewedNeighbor, hexSize);
    float minDist2 = min(minDists.y, max(minDists.x, dist));
    float minDist1 = min(minDists.x, dist);
    return vec2(minDist1, minDist2);
}

vec2 minDistancesFromPoint(vec3 p, float hexSize, float agitation) {
    vec3 centreCoord = hexCoord(p, hexSize);
    vec2 minDists = vec2(2.0, 2.0);

    // middle layer
    float minXY = -2.0;
    float maxXY = 2.0;
    float z = 0.0;
    for (float x=minXY; x<=maxXY; x++) {
        for (float y=minXY; y<=maxXY; y++) {
            if (abs(x + y) <= 2.0) {
                minDists = minDistances(p, centreCoord, vec3(x, y, z), minDists, hexSize, agitation);
            }
        }
    }

    // 1 layer above/below
    for (float z=-1.0; z<=1.0; z+=2.0) {
        float signZ = sign(z);
        float halfSignZ = signZ / 2.0;
        minXY = -1.5 - halfSignZ;
        maxXY = 1.5 - halfSignZ;
        for (float x=minXY; x<=maxXY; x++) {
            for (float y=minXY; y<=maxXY; y++) {
                if (!(
                (x == signZ * 1.0 && y == signZ * 1.0) ||
                (x == signZ * -2.0 && y == signZ * -1.0) ||
                (x == signZ * -1.0 && y == signZ * -2.0) ||
                (x == signZ * -2.0 && y == signZ * -2.0)
                )) {
                    minDists = minDistances(p, centreCoord, vec3(x, y, z), minDists, hexSize, agitation);
                }
            }
        }
    }

    // 2 layers above/below
    for (float z=-2.0; z<=2.0; z+=4.0) {
        float signZ = sign(z);
        minXY = -1.0 - signZ;
        maxXY = 1.0 - signZ;
        for (float x=minXY; x<=maxXY; x++) {
            for (float y=minXY; y<=maxXY; y++) {
                if (abs(x + y) <= 2.0) {
                    minDists = minDistances(p, centreCoord, vec3(x, y, z), minDists, hexSize, agitation);
                }
            }
        }
    }

    return minDists;
}



void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float hexSize = 0.025; // max 1.0 - one hex cell will fill the screen
    float agitation = 1.0; // max 1.0 - otherwise artifacts will appear
    float speed = 0.025;

    vec2 xy = vec2(fragCoord.xy / iResolution.x);
    vec3 p = vec3(xy, iTime * speed);
    vec2 minDists = minDistancesFromPoint(p, hexSize, agitation);
    // minDists.x: smallest distance to centre of other cell
    // minDists.y: second smallest distance to centre of other cell

    float gray = 1.0 - (minDists.y - minDists.x);
    gray *= gray;
    fragColor = vec4(vec3(gray), 1.0);
}
