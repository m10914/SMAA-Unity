Shader "Custom/SMAAshader" {
	Properties {
		_SrcTex ("source texture", 2D) = "white" {}
		_MainTex ("Base (RGB)", 2D) = "white" {}
		areaTex ("area texture", 2D) = "white" {}
		luminTex ("lumin texture", 2D) = "white" {}
		searchTex ("search texture", 2D) = "white" {}
	}
	SubShader {
		Tags { "RenderType"="Opaque" }
		LOD 200
		
		/*
		VertexShader = compile vs_3_0 DX9_SMAAEdgeDetectionVS();
        PixelShader = compile ps_3_0 DX9_SMAAColorEdgeDetectionPS();
        ZEnable = false;        
        SRGBWriteEnable = false;
        AlphaBlendEnable = false;
        AlphaTestEnable = false;

        // We will be creating the stencil buffer for later usage.
        StencilEnable = true;
        StencilPass = REPLACE;
        StencilRef = 1;
		*/
		//-----------------------------------------------------
		// Pass 0 - colorEdgeDetection
		//-----------------------------------------------------		
		Pass
		{
			ZWrite Off
			ZTest Always

			CGPROGRAM
			#pragma fragment fragment
			#pragma vertex vertex
			#include "UnityCG.cginc"
			#pragma target 3.0

			#define mad(a, b, c) (a * b + c)
			#define SMAA_RT_METRICS float4(1.0 / 1280.0, 1.0 / 720.0, 1280.0, 720.0)
			#define SMAA_THRESHOLD 0.05
			#define SMAA_MAX_SEARCH_STEPS 32
			#define SMAA_MAX_SEARCH_STEPS_DIAG 16
			#define SMAA_CORNER_ROUNDING 25
			#define SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR 2.0

			sampler2D _MainTex;

			struct vertOUT
			{
				float4 position: POSITION;
				float2 texcoord: TEXCOORD0;
				float4 offset0: TEXCOORD1;
				float4 offset1: TEXCOORD2;
				float4 offset2: TEXCOORD3;
			};

			vertOUT vertex (appdata_base IN) 
			{ 
				vertOUT OUT;

				OUT.position = mul(UNITY_MATRIX_MVP, IN.vertex);
				OUT.texcoord = IN.texcoord.xy;

				OUT.offset0 = mad(SMAA_RT_METRICS.xyxy, float4(-1.0, 0.0, 0.0, -1.0), IN.texcoord.xyxy);
				OUT.offset1 = mad(SMAA_RT_METRICS.xyxy, float4( 1.0, 0.0, 0.0,  1.0), IN.texcoord.xyxy);
				OUT.offset2 = mad(SMAA_RT_METRICS.xyxy, float4(-2.0, 0.0, 0.0, -2.0), IN.texcoord.xyxy);

				return OUT; 
			}

			float4 fragment ( vertOUT IN ) : COLOR
			{
				// Calculate the threshold:
				float2 threshold = float2(SMAA_THRESHOLD, SMAA_THRESHOLD);

				// Calculate color deltas:
				float4 delta;
				float3 C = tex2D(_MainTex, IN.texcoord).rgb;

				float3 Cleft = tex2D(_MainTex, IN.offset0.xy).rgb;
				float3 t = abs(C - Cleft);
				delta.x = max(max(t.r, t.g), t.b);

				float3 Ctop  = tex2D(_MainTex, IN.offset0.zw).rgb;
				t = abs(C - Ctop);
				delta.y = max(max(t.r, t.g), t.b);

				// We do the usual threshold:
				float2 edges = step(threshold, delta.xy);

				// Then discard if there is no edge:
				if (dot(edges, float2(1.0, 1.0)) == 0.0)
					discard;

				// Calculate right and bottom deltas:
				float3 Cright = tex2D(_MainTex, IN.offset1.xy).rgb;
				t = abs(C - Cright);
				delta.z = max(max(t.r, t.g), t.b);

				float3 Cbottom  = tex2D(_MainTex, IN.offset1.zw).rgb;
				t = abs(C - Cbottom);
				delta.w = max(max(t.r, t.g), t.b);

				// Calculate the maximum delta in the direct neighborhood:
				float2 maxDelta = max(delta.xy, delta.zw);

				// Calculate left-left and top-top deltas:
				float3 Cleftleft  = tex2D(_MainTex, IN.offset2.xy).rgb;
				t = abs(C - Cleftleft);
				delta.z = max(max(t.r, t.g), t.b);

				float3 Ctoptop = tex2D(_MainTex, IN.offset2.zw).rgb;
				t = abs(C - Ctoptop);
				delta.w = max(max(t.r, t.g), t.b);

				// Calculate the final maximum delta:
				maxDelta = max(maxDelta.xy, delta.zw);
				maxDelta = max(maxDelta.xx, maxDelta.yy);

				// Local contrast adaptation:
				edges.xy *= step((1.0 / SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR) * maxDelta, delta.xy);

				return float4(edges, 0.0, 0.0);
			}

			ENDCG

		}


		/*
		VertexShader = compile vs_3_0 DX9_SMAABlendingWeightCalculationVS();
        PixelShader = compile ps_3_0 DX9_SMAABlendingWeightCalculationPS();
        ZEnable = false;
        SRGBWriteEnable = false;
        AlphaBlendEnable = false;
        AlphaTestEnable = false;

        // Here we want to process only marked pixels.
        StencilEnable = true;
        StencilPass = KEEP;
        StencilFunc = EQUAL;
        StencilRef = 1;
		*/
		//--------------------------------------
		// Pass 1 - blend weights
		// Order: second
		//--------------------------------------
		Pass
		{
			ZWrite Off
			ZTest Always


			CGPROGRAM
			#pragma fragment fragment
			#pragma vertex vertex
			#include "UnityCG.cginc"
			#pragma target 3.0
			#pragma glsl

			#define mad(a, b, c) (a * b + c)
			#define SMAA_RT_METRICS float4(1.0 / 1280.0, 1.0 / 720.0, 1280.0, 720.0)
			#define SMAA_THRESHOLD 0.05
			#define SMAA_MAX_SEARCH_STEPS 32
			#define SMAA_MAX_SEARCH_STEPS_DIAG 16
			#define SMAA_CORNER_ROUNDING 25
			#define SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR 2.0
			#define SMAA_AREATEX_MAX_DISTANCE 16
			#define SMAA_AREATEX_MAX_DISTANCE_DIAG 20
			#define SMAASampleLevelZeroOffset(tex, coord, offset) tex2Dlod(tex, float4(coord + offset * SMAA_RT_METRICS.xy, 0.0, 0.0))
			#define SMAA_AREATEX_PIXEL_SIZE (1.0 / float2(160.0, 560.0))
			#define SMAA_AREATEX_SUBTEX_SIZE (1.0 / 7.0)
			#define SMAA_AREATEX_SELECT(sample) sample.rg

			sampler2D _MainTex;
			sampler2D areaTex;
			sampler2D searchTex;
			sampler2D luminTex;

//--------------------------------------------------------
// S M A A   framework

float SMAASearchLength(float2 e, float offset)
{
    // We have cropped the texture, so we need to unpack the coordinates:
    float2 ratio = float2(66.0 / 64.0, 33.0 / 16.0);

    // The texture is flipped vertically, with left and right cases taking half
    // of the space horizontally:
    float2 scale = ratio * float2(0.5, -1.0);

    // We need to offset the coordinates, depending on left/right cases:
    float2 bias = ratio * float2(offset, 1.0);

    return tex2Dlod(searchTex, float4(mad(scale, e, bias),0,0)).a;
}

float SMAASearchXLeft(float2 texcoord, float end)
{
	float2 e = float2(0.0, 1.0);
	while (texcoord.x > end && 
			e.g > 0.8281 && // Is there some edge not activated?
			e.r == 0.0) { // Or is there a crossing edge that breaks the line?
		e = tex2Dlod(_MainTex, float4(texcoord,0,0)).rg;
		texcoord = mad(-float2(2.0, 0.0), SMAA_RT_METRICS.xy, texcoord);
	}

	float offset = mad(-(255.0 / 127.0), SMAASearchLength(e, 0.0), 3.25);
	return mad(SMAA_RT_METRICS.x, offset, texcoord.x);
}

float SMAASearchYUp(float2 texcoord, float end) {
    float2 e = float2(1.0, 0.0);
    while (texcoord.y > end && 
           e.r > 0.8281 && // Is there some edge not activated?
           e.g == 0.0) { // Or is there a crossing edge that breaks the line?
        e = tex2Dlod(_MainTex, float4(texcoord,0,0)).rg;
        texcoord = mad(-float2(0.0, 2.0), SMAA_RT_METRICS.xy, texcoord);
    }
    float offset = mad(-(255.0 / 127.0), SMAASearchLength(e.gr, 0.0), 3.25);
    return mad(SMAA_RT_METRICS.y, offset, texcoord.y);
}

float SMAASearchXRight(float2 texcoord, float end) {
    float2 e = float2(0.0, 1.0);
    while (texcoord.x < end && 
           e.g > 0.8281 && // Is there some edge not activated?
           e.r == 0.0) { // Or is there a crossing edge that breaks the line?
        e = tex2Dlod(_MainTex, float4(texcoord,0,0)).rg;
        texcoord = mad(float2(2.0, 0.0), SMAA_RT_METRICS.xy, texcoord);
    }
    float offset = mad(-(255.0 / 127.0), SMAASearchLength(e, 0.5), 3.25);
    return mad(-SMAA_RT_METRICS.x, offset, texcoord.x);
}

float SMAASearchYDown(float2 texcoord, float end) {
    float2 e = float2(1.0, 0.0);
    while (texcoord.y < end && 
           e.r > 0.8281 && // Is there some edge not activated?
           e.g == 0.0) { // Or is there a crossing edge that breaks the line?
        e = tex2Dlod(_MainTex, float4(texcoord,0,0)).rg;
        texcoord = mad(float2(0.0, 2.0), SMAA_RT_METRICS.xy, texcoord);
    }
    float offset = mad(-(255.0 / 127.0), SMAASearchLength(e.gr, 0.5), 3.25);
    return mad(-SMAA_RT_METRICS.y, offset, texcoord.y);
}

float2 SMAAArea(float2 dist, float e1, float e2, float offset) {

    // Rounding prevents precision errors of bilinear filtering:
    float2 texcoord = mad(float(SMAA_AREATEX_MAX_DISTANCE), round(4.0 * float2(e1, e2)), dist);
    
    // We do a scale and bias for mapping to texel space:
    texcoord = mad(SMAA_AREATEX_PIXEL_SIZE, texcoord, 0.5 * SMAA_AREATEX_PIXEL_SIZE);

    // Move to proper place, according to the subpixel offset:
    texcoord.y = mad(SMAA_AREATEX_SUBTEX_SIZE, offset, texcoord.y);

    // Do it!
    return float2(
		//tex2D(areaTex, texcoord).a,
		//tex2D(luminTex, texcoord).a);
		tex2Dlod(areaTex, float4(texcoord,0,0)).a,
		tex2Dlod(luminTex, float4(texcoord,0,0)).a);
}
void SMAAMovc(float2 cond, inout float2 variable, float2 value) {
    if (cond.x) variable.x = value.x;
    if (cond.y) variable.y = value.y;
}

void SMAAMovc(float4 cond, inout float4 variable, float4 value) {
    SMAAMovc(cond.xy, variable.xy, value.xy);
    SMAAMovc(cond.zw, variable.zw, value.zw);
}
float2 SMAADecodeDiagBilinearAccess(float2 e) {
    e.r = e.r * abs(5.0 * e.r - 5.0 * 0.75);
    return round(e);
}
float4 SMAADecodeDiagBilinearAccess(float4 e) {
    e.rb = e.rb * abs(5.0 * e.rb - 5.0 * 0.75);
    return round(e);
}
float2 SMAASearchDiag1(float2 texcoord, float2 dir, out float2 e) {
    float4 coord = float4(texcoord, -1.0, 1.0);
    while (coord.z < float(SMAA_MAX_SEARCH_STEPS_DIAG - 1) &&
           coord.w > 0.9) {
        coord.xyz = mad(float3(SMAA_RT_METRICS.xy, 1.0), float3(dir, 1.0), coord.xyz);
        e = tex2Dlod(_MainTex, float4(coord.xy,0,0)).rg;
        coord.w = dot(e, 0.5);
    }
    return coord.zw;
}

float2 SMAASearchDiag2(float2 texcoord, float2 dir, out float2 e) {
    float4 coord = float4(texcoord, -1.0, 1.0);
    coord.x += 0.25 * SMAA_RT_METRICS.x; // See @SearchDiag2Optimization
    while (coord.z < float(SMAA_MAX_SEARCH_STEPS_DIAG - 1) &&
           coord.w > 0.9) {
        coord.xyz = mad(float3(SMAA_RT_METRICS.xy, 1.0), float3(dir, 1.0), coord.xyz);

        // @SearchDiag2Optimization
        // Fetch both edges at once using bilinear filtering:
        e = tex2Dlod(_MainTex, float4(coord.xy,0,0)).rg;
        e = SMAADecodeDiagBilinearAccess(e);

        coord.w = dot(e, 0.5);
    }
    return coord.zw;
}

float2 SMAAAreaDiag(float2 dist, float2 e, float offset) {
    float2 texcoord = mad(float(SMAA_AREATEX_MAX_DISTANCE_DIAG), e, dist);

    // We do a scale and bias for mapping to texel space:
    texcoord = mad(SMAA_AREATEX_PIXEL_SIZE, texcoord, 0.5 * SMAA_AREATEX_PIXEL_SIZE);

    // Diagonal areas are on the second half of the texture:
    texcoord.x += 0.5;

    // Move to proper place, according to the subpixel offset:
    texcoord.y += SMAA_AREATEX_SUBTEX_SIZE * offset;

    // Do it!
    return float2(
		tex2Dlod(areaTex, float4(texcoord,0,0)).a,
		tex2Dlod(luminTex, float4(texcoord,0,0)).a);
}

float2 SMAACalculateDiagWeights(float2 texcoord, float2 e, float4 subsampleIndices) {
    float2 weights = float2(0.0, 0.0);

    // Search for the line ends:
    float4 d;
    float2 end;
    if (e.r > 0.0) {
        d.xz = SMAASearchDiag1(texcoord, float2(-1.0,  1.0), end);
        d.x += float(end.y > 0.9);
    } else
        d.xz = 0.0;
    d.yw = SMAASearchDiag1(texcoord, float2(1.0, -1.0), end);

    if (d.x + d.y > 2.0) { // d.x + d.y + 1 > 3
        // Fetch the crossing edges:
        float4 coords = mad(float4(-d.x + 0.25, d.x, d.y, -d.y - 0.25), SMAA_RT_METRICS.xyxy, texcoord.xyxy);
        float4 c;
        c.xy = SMAASampleLevelZeroOffset(_MainTex, coords.xy, int2(-1,  0)).rg;
        c.zw = SMAASampleLevelZeroOffset(_MainTex, coords.zw, int2( 1,  0)).rg;
        c.yxwz = SMAADecodeDiagBilinearAccess(c.xyzw);

        // Merge crossing edges at each side into a single value:
        float2 cc = mad(2.0, c.xz, c.yw);

        // Remove the crossing edge if we didn't found the end of the line:
        SMAAMovc(step(0.9, d.zw), cc, 0.0);

        // Fetch the areas for this line:
        weights += SMAAAreaDiag(d.xy, cc, subsampleIndices.z);
    }

    // Search for the line ends:
    d.xz = SMAASearchDiag2(texcoord, float2(-1.0, -1.0), end);
    if (SMAASampleLevelZeroOffset(_MainTex, texcoord, int2(1, 0)).r > 0.0) {
        d.yw = SMAASearchDiag2(texcoord, float2(1.0, 1.0), end);
        d.y += float(end.y > 0.9);
    } else
        d.yw = 0.0;

    if (d.x + d.y > 2.0) { // d.x + d.y + 1 > 3
        // Fetch the crossing edges:
        float4 coords = mad(float4(-d.x, -d.x, d.y, d.y), SMAA_RT_METRICS.xyxy, texcoord.xyxy);
        float4 c;
        c.x  = SMAASampleLevelZeroOffset(_MainTex, coords.xy, int2(-1,  0)).g;
        c.y  = SMAASampleLevelZeroOffset(_MainTex, coords.xy, int2( 0, -1)).r;
        c.zw = SMAASampleLevelZeroOffset(_MainTex, coords.zw, int2( 1,  0)).gr;
        float2 cc = mad(2.0, c.xz, c.yw);

        // Remove the crossing edge if we didn't found the end of the line:
        SMAAMovc(step(0.9, d.zw), cc, 0.0);

        // Fetch the areas for this line:
        weights += SMAAAreaDiag(d.xy, cc, subsampleIndices.w).gr;
    }

    return weights;
}


float2 SMAADetectHorizontalCornerPattern(float2 weights, float2 texcoord, float2 d) {

    float4 coords = mad(float4(d.x, 0.0, d.y, 0.0),
                        SMAA_RT_METRICS.xyxy, texcoord.xyxy);
    float2 e;
    e.r = SMAASampleLevelZeroOffset(_MainTex, coords.xy, int2(0.0,  1.0)).r;
    bool left = abs(d.x) < abs(d.y);
    e.g = SMAASampleLevelZeroOffset(_MainTex, coords.xy, int2(0.0, -2.0)).r;
    if (left) weights *= saturate(float(SMAA_CORNER_ROUNDING) / 100.0 + 1.0 - e);

    e.r = SMAASampleLevelZeroOffset(_MainTex, coords.zw, int2(1.0,  1.0)).r;
    e.g = SMAASampleLevelZeroOffset(_MainTex, coords.zw, int2(1.0, -2.0)).r;
    if (!left) weights *= saturate(float(SMAA_CORNER_ROUNDING) / 100.0 + 1.0 - e);

	return weights;
}

float2 SMAADetectVerticalCornerPattern(float2 weights, float2 texcoord, float2 d) {

    float4 coords = mad(float4(0.0, d.x, 0.0, d.y),
                        SMAA_RT_METRICS.xyxy, texcoord.xyxy);
    float2 e;
    e.r = SMAASampleLevelZeroOffset(_MainTex, coords.xy, int2( 1.0, 0.0)).g;
    bool left = abs(d.x) < abs(d.y);
    e.g = SMAASampleLevelZeroOffset(_MainTex, coords.xy, int2(-2.0, 0.0)).g;
    if (left) weights *= saturate(float(SMAA_CORNER_ROUNDING) / 100.0 + 1.0 - e);

    e.r = SMAASampleLevelZeroOffset(_MainTex, coords.zw, int2( 1.0, 1.0)).g;
    e.g = SMAASampleLevelZeroOffset(_MainTex, coords.zw, int2(-2.0, 1.0)).g;
    if (!left) weights *= saturate(float(SMAA_CORNER_ROUNDING) / 100.0 + 1.0 - e);

	return weights;
}
//--------------------------------------------------------

			struct vertOUT
			{
				float4 position : POSITION;
				float2 texcoord: TEXCOORD0;
				float4 offset0: TEXCOORD1;
				float4 offset1: TEXCOORD2;
				float4 offset2: TEXCOORD3;
				float2 pixcoord: TEXCOORD4;
			};

			vertOUT vertex (appdata_base IN) 
			{ 
				vertOUT OUT;

				OUT.position = mul(UNITY_MATRIX_MVP, IN.vertex);
				OUT.texcoord = IN.texcoord.xy;

				OUT.pixcoord = IN.texcoord * SMAA_RT_METRICS.zw;

				OUT.offset0 = mad(SMAA_RT_METRICS.xyxy, float4(-0.25, -0.125,  1.25, -0.125), IN.texcoord.xyxy);
				OUT.offset1 = mad(SMAA_RT_METRICS.xyxy, float4(-0.125, -0.25, -0.125,  1.25), IN.texcoord.xyxy);
				OUT.offset2 = mad(SMAA_RT_METRICS.xxyy,
                    float4(-2.0, 2.0, -2.0, 2.0) * float(SMAA_MAX_SEARCH_STEPS),
                    float4(OUT.offset0.xz, OUT.offset1.yw));

				return OUT; 
			}

			float4 fragment ( vertOUT IN ) : COLOR
			{
				float4 subsampleIndices = float4(0, 0, 0, 0);

				float4 weights = float4(0.0, 0.0, 0.0, 0.0);

				float2 e = tex2Dlod(_MainTex, float4(IN.texcoord,0,0)).rg;

				if (e.g > 0.0)
				{ // Edge at north

					//search for diagonales
					//weights.rg = SMAACalculateDiagWeights(IN.texcoord, e, subsampleIndices);

					//if (dot(weights.rg, float2(1.0, 1.0)) == 0.0)
					//{
						float2 d;

						// Find the distance to the left:
						float2 coords;
						coords.x = SMAASearchXLeft(IN.offset0.xy, IN.offset2.x);
						coords.y = IN.offset1.y; // offset[1].y = texcoord.y - 0.25 * SMAA_RT_METRICS.y (@CROSSING_OFFSET)
						d.x = coords.x;

						// Now fetch the left crossing edges, two at a time using bilinear
						// filtering. Sampling at -0.25 (see @CROSSING_OFFSET) enables to
						// discern what value each edge has:
						float e1 = tex2Dlod(_MainTex, float4(coords,0,0)).r;

						// Find the distance to the right:
						coords.x = SMAASearchXRight(IN.offset0.zw, IN.offset2.y);
						d.y = coords.x;

						// We want the distances to be in pixel units (doing this here allow to
						// better interleave arithmetic and memory accesses):
						d = mad(SMAA_RT_METRICS.z, d, -IN.pixcoord.x);

						// SMAAArea below needs a sqrt, as the areas texture is compressed
						// quadratically:
						float2 sqrt_d = sqrt(abs(d));

						// Fetch the right crossing edges:
						float e2 = SMAASampleLevelZeroOffset(_MainTex, coords, int2(1, 0)).r;

						// Ok, we know how this pattern looks like, now it is time for getting
						// the actual area:
						weights.rg = SMAAArea(sqrt_d, e1, e2, subsampleIndices.y);

						// Fix corners:
						//weights.rg = SMAADetectHorizontalCornerPattern(weights.rg, IN.texcoord, d);;		
					//}
					//else e.r = 0.0;

				}

				if (e.r > 0.0) { // Edge at west

					float2 d;

					// Find the distance to the top:
					float2 coords;
					coords.y = SMAASearchYUp(IN.offset1.xy, IN.offset2.z);
					coords.x = IN.offset0.x; // offset[1].x = texcoord.x - 0.25 * SMAA_RT_METRICS.x;
					d.x = coords.y;

					// Fetch the top crossing edges:
					float e1 = tex2Dlod(_MainTex, float4(coords,0,0)).g;

					// Find the distance to the bottom:
					coords.y = SMAASearchYDown(IN.offset1.zw, IN.offset2.w);
					d.y = coords.y;

					// We want the distances to be in pixel units:
					d = mad(SMAA_RT_METRICS.w, d, -IN.pixcoord.y);

					// SMAAArea below needs a sqrt, as the areas texture is compressed 
					// quadratically:
					float2 sqrt_d = sqrt(abs(d));

					// Fetch the bottom crossing edges:
					float e2 = SMAASampleLevelZeroOffset(_MainTex, coords, int2(0, 1)).g;

					// Get the area for this direction:
					weights.ba = SMAAArea(sqrt_d, e1, e2, subsampleIndices.x);

					// Fix corners:
					//weights.ba = SMAADetectVerticalCornerPattern(weights.ba, IN.texcoord, d);
				}

				return weights;
			}

			ENDCG

		}


		//-----------------------------------------
		// Pass 2
		// Neighborhood blend
		// Order: last
		//-----------------------------------------
		Pass
		{
			ZWrite Off
			ZTest Always


			CGPROGRAM
			#pragma fragment fragment
			#pragma vertex vertex
			#include "UnityCG.cginc"
			#pragma target 3.0
			#pragma glsl

			#define mad(a, b, c) (a * b + c)
			#define SMAA_RT_METRICS float4(1.0 / 1280.0, 1.0 / 720.0, 1280.0, 720.0)
			#define SMAA_THRESHOLD 0.05
			#define SMAA_MAX_SEARCH_STEPS 32
			#define SMAA_MAX_SEARCH_STEPS_DIAG 16
			#define SMAA_CORNER_ROUNDING 25
			#define SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR 2.0

			sampler2D _MainTex;
			sampler2D _SrcTex;

//-------------------------------------------------
// S M A A  framework

void SMAAMovc(float2 cond, inout float2 variable, float2 value) {
    if (cond.x) variable.x = value.x;
    if (cond.y) variable.y = value.y;
}

void SMAAMovc(float4 cond, inout float4 variable, float4 value) {
    SMAAMovc(cond.xy, variable.xy, value.xy);
    SMAAMovc(cond.zw, variable.zw, value.zw);
}

//--------------------------------------------------

			struct vertOUT
			{
				float4 position : POSITION;
				float2 texcoord: TEXCOORD0;
				float4 offset: TEXCOORD1;
			};

			vertOUT vertex (appdata_base IN) 
			{ 
				vertOUT OUT;

				OUT.position = mul(UNITY_MATRIX_MVP, IN.vertex);
				OUT.texcoord = IN.texcoord.xy;

				OUT.offset = mad(SMAA_RT_METRICS.xyxy, float4( 1.0, 0.0, 0.0,  1.0), IN.texcoord.xyxy);

				return OUT; 
			}

			float4 fragment ( vertOUT IN ) : COLOR
			{
				// Fetch the blending weights for current pixel:
				float4 a;
				a.x = tex2Dlod(_MainTex, float4(IN.offset.xy,0,0)).a; // Right
				a.y = tex2Dlod(_MainTex, float4(IN.offset.zw,0,0)).g; // Top
				a.wz = tex2Dlod(_MainTex, float4(IN.texcoord,0,0)).xz; // Bottom / Left

				// Is there any blending weight with a value greater than 0.0?
				if (dot(a, float4(1.0, 1.0, 1.0, 1.0)) < 1e-5)
				{
					float4 color = tex2Dlod(_SrcTex, float4(IN.texcoord,0,0));
					return color;
				}
				else
				{
					bool horizontal = max(a.x, a.z) > max(a.y, a.w); // max(horizontal) > max(vertical)

					// Calculate the blending offsets:
					float4 blendingOffset = float4(0.0, a.y, 0.0, a.w);
					float2 blendingWeight = a.yw;
					SMAAMovc(horizontal, blendingOffset, float4(a.x, 0.0, a.z, 0.0));
					SMAAMovc(horizontal, blendingWeight, a.xz);
					blendingWeight /= dot(blendingWeight, 1.0);

					// Calculate the texture coordinates:
					float4 blendingCoord = mad(blendingOffset, float4(SMAA_RT_METRICS.xy, -SMAA_RT_METRICS.xy), IN.texcoord.xyxy);

					// We exploit bilinear filtering to mix current pixel with the chosen
					// neighbor:
					float4 color = blendingWeight.x * tex2Dlod(_SrcTex, float4(blendingCoord.xy,0,0));
					color += blendingWeight.y * tex2Dlod(_SrcTex, float4(blendingCoord.zw,0,0));

					return color;
				}
			}

			ENDCG
		}



		//---------------------------------------------------------------------------
		// additionl passes - luma

		//-----------------------------------------------------
		// Pass 3 - lumaEdgeDetection
		//-----------------------------------------------------		
		Pass
		{
			ZWrite Off
			ZTest Always

			CGPROGRAM
			#pragma fragment fragment
			#pragma vertex vertex
			#include "UnityCG.cginc"
			#pragma target 3.0
			#pragma glsl

			#define mad(a, b, c) (a * b + c)
			#define SMAA_RT_METRICS float4(1.0 / 1280.0, 1.0 / 720.0, 1280.0, 720.0)
			#define SMAA_THRESHOLD 0.05
			#define SMAA_MAX_SEARCH_STEPS 32
			#define SMAA_MAX_SEARCH_STEPS_DIAG 16
			#define SMAA_CORNER_ROUNDING 25
			#define SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR 2.0

			sampler2D _MainTex;

			struct vertOUT
			{
				float4 position: POSITION;
				float2 texcoord: TEXCOORD0;
				float4 offset0: TEXCOORD1;
				float4 offset1: TEXCOORD2;
				float4 offset2: TEXCOORD3;
			};

			vertOUT vertex (appdata_base IN) 
			{ 
				vertOUT OUT;

				OUT.position = mul(UNITY_MATRIX_MVP, IN.vertex);
				OUT.texcoord = IN.texcoord.xy;

				OUT.offset0 = mad(SMAA_RT_METRICS.xyxy, float4(-1.0, 0.0, 0.0, -1.0), IN.texcoord.xyxy);
				OUT.offset1 = mad(SMAA_RT_METRICS.xyxy, float4( 1.0, 0.0, 0.0,  1.0), IN.texcoord.xyxy);
				OUT.offset2 = mad(SMAA_RT_METRICS.xyxy, float4(-2.0, 0.0, 0.0, -2.0), IN.texcoord.xyxy);

				return OUT; 
			}

			float4 fragment ( vertOUT IN ) : COLOR
			{
				// Calculate the threshold:
				float2 threshold = float2(SMAA_THRESHOLD, SMAA_THRESHOLD);

				// Calculate lumas:
				float3 weights = float3(0.2126, 0.7152, 0.0722);
				float L = dot(tex2Dlod(_MainTex, float4(IN.texcoord,0,0)).rgb, weights);

				float Lleft = dot(tex2Dlod(_MainTex, float4(IN.offset0.xy,0,0)).rgb, weights);
				float Ltop  = dot(tex2Dlod(_MainTex, float4(IN.offset0.zw,0,0)).rgb, weights);

				// We do the usual threshold:
				float4 delta;
				delta.xy = abs(L - float2(Lleft, Ltop));
				float2 edges = step(threshold, delta.xy);

				// Then discard if there is no edge:
				if (dot(edges, float2(1.0, 1.0)) == 0.0)
					discard;

				// Calculate right and bottom deltas:
				float Lright = dot(tex2Dlod(_MainTex, float4(IN.offset1.xy,0,0)).rgb, weights);
				float Lbottom  = dot(tex2Dlod(_MainTex, float4(IN.offset1.zw,0,0)).rgb, weights);
				delta.zw = abs(L - float2(Lright, Lbottom));

				// Calculate the maximum delta in the direct neighborhood:
				float2 maxDelta = max(delta.xy, delta.zw);

				// Calculate left-left and top-top deltas:
				float Lleftleft = dot(tex2Dlod(_MainTex, float4(IN.offset2.xy,0,0)).rgb, weights);
				float Ltoptop = dot(tex2Dlod(_MainTex, float4(IN.offset2.zw,0,0)).rgb, weights);
				delta.zw = abs(float2(Lleft, Ltop) - float2(Lleftleft, Ltoptop));

				// Calculate the final maximum delta:
				maxDelta = max(maxDelta.xy, delta.zw);
				maxDelta = max(maxDelta.xx, maxDelta.yy);

				// Local contrast adaptation:
				edges.xy *= step((1.0 / SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR) * maxDelta, delta.xy);

				return float4(edges, 0.0, 0.0);
			}

			ENDCG

		}

	} 
	FallBack "Diffuse"
}
