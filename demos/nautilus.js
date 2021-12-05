import PicoGL from "../node_modules/picogl/build/module/picogl.js";
import {mat4, vec3, quat} from "../node_modules/gl-matrix/esm/index.js";

import {positions, normals, indices, uvs} from "../blender/nautilus.js"

let postPositions = new Float32Array([
    0.0, 1.0,
    1.0, 1.0,
    0.0, 0.0,
    1.0, 0.0,
]);

let postIndices = new Uint32Array([
    0, 2, 1,
    2, 3, 1
]);


// ******************************************************
// **               Light configuration                **
// ******************************************************

let ambientLightColor = vec3.fromValues(0.05, 0.05, 0.1);
let numberOfLights = 2;
let lightColors = [vec3.fromValues(1.0, 0.5, 0.7), vec3.fromValues(0.5, 0.6, 1.0)];
let lightInitialPositions = [vec3.fromValues(5, 0, 2), vec3.fromValues(-5, 0, 2)];
let lightPositions = [vec3.create(), vec3.create()];


// language=GLSL
let lightCalculationShader = `
    uniform vec3 cameraPosition;
    uniform vec3 ambientLightColor;    
    uniform vec3 lightColors[${numberOfLights}];        
    uniform vec3 lightPositions[${numberOfLights}];
    
    // This function calculates light reflection using Phong reflection model (ambient + diffuse + specular)
    vec4 calculateLights(vec3 normal, vec3 position) {
        vec3 viewDirection = normalize(cameraPosition.xyz - position);
        vec4 color = vec4(ambientLightColor, 1.0);
                
        for (int i = 0; i < lightPositions.length(); i++) {
            vec3 lightDirection = normalize(lightPositions[i] - position);
            
            // Lambertian reflection (ideal diffuse of matte surfaces) is also a part of Phong model                        
            float diffuse = max(dot(lightDirection, normal), 0.0);                                    
                      
            // Phong specular highlight 
            float specular = pow(max(dot(viewDirection, reflect(-lightDirection, normal)), 0.0), 10.0);
            
            // Blinn-Phong improved specular highlight                        
            //float specular = pow(max(dot(normalize(lightDirection + viewDirection), normal), 0.0), 200.0);
            
            color.rgb += lightColors[i] * diffuse + specular;
        }
        return color;
    }
`;

// language=GLSL
let fragmentShader = `
    #version 300 es
    precision highp float;        
    ${lightCalculationShader}

    uniform sampler2D tex;
    uniform samplerCube cubemap;

    in vec3 viewDir;
    in vec3 vPosition;    
    in vec3 vNormal;
    in vec4 vColor;
    in vec2 v_uv;
    
    out vec4 outColor;        
    
    void main() {
        vec3 reflectedDir = reflect(viewDir, normalize(vNormal));        
        
        // For Phong shading (per-fragment) move color calculation from vertex to fragment shader
        outColor = calculateLights(normalize(vNormal), vPosition) * texture(tex, v_uv);
        // vec4 reflection = pow(texture(cubemap, reflectedDir), vec4(5.0)) * 0.3;
        // outColor += reflection;
        // outColor = vColor;
    }
`;

// language=GLSL
let vertexShader = `
    #version 300 es
    ${lightCalculationShader}
        
    layout(location=0) in vec4 position;
    layout(location=1) in vec4 normal;
    layout(location=2) in vec2 uv;
    
    uniform mat4 viewProjectionMatrix;
    uniform mat4 modelMatrix;

    out vec3 viewDir;
    out vec3 vPosition;    
    out vec3 vNormal;
    out vec4 vColor;
    out vec2 v_uv;
    
    void main() {
        vec4 worldPosition = modelMatrix * position;
        
        vPosition = worldPosition.xyz;        
        vNormal = (modelMatrix * normal).xyz;
        v_uv = vec2(uv.x, -uv.y);
        viewDir = (modelMatrix * position).xyz - cameraPosition;
        
        // For Gouraud shading (per-vertex) move color calculation from fragment to vertex shader
        //vColor = calculateLights(normalize(vNormal), vPosition);
        
        gl_Position = viewProjectionMatrix * worldPosition;                        
    }
`;

// language=GLSL
let postFragmentShader = `
    #version 300 es
    precision mediump float;
    
    uniform sampler2D tex;
    uniform sampler2D depthTex;
    uniform float time;
    uniform sampler2D noiseTex;
    
    in vec4 v_position;
    
    out vec4 outColor;
    
    vec4 depthOfField(vec4 col, float depth, vec2 uv) {
        vec4 blur = vec4(0.0);
        float n = 0.0;
        for (float u = -1.0; u <= 1.0; u += 0.4)    
            for (float v = -1.0; v <= 1.0; v += 0.4) {
                float factor = abs(depth - 0.995) * 350.0;
                blur += texture(tex, uv + vec2(u, v) * factor * 0.02);
                n += 1.0;
            }                
        return blur / n;
    }
    
    vec4 ambientOcclusion(vec4 col, float depth, vec2 uv) {
        if (depth == 1.0) return col;
        for (float u = -2.0; u <= 2.0; u += 0.4)    
            for (float v = -2.0; v <= 2.0; v += 0.4) {                
                float d = texture(depthTex, uv + vec2(u, v) * 0.01).r;
                if (d != 1.0) {
                    float diff = abs(depth - d);
                    col *= 1.0 - diff * 30.0;
                }
            }
        return col;        
    }   
    
    float random(vec2 seed) {
        return texture(noiseTex, seed * 5.0 + sin(time * 543.12) * 54.12).r - 0.5;
    }
    
    void main() {
        vec4 col = texture(tex, v_position.xy);
        float depth = texture(depthTex, v_position.xy).r;
        
        // Chromatic aberration 
        //vec2 caOffset = vec2(0.01, 0.0);
        //col.r = texture(tex, v_position.xy - caOffset).r;
        //col.b = texture(tex, v_position.xy + caOffset).b;
        
        // Depth of field
        col = depthOfField(col, depth, v_position.xy);

        // Noise         
        col.rgb += (2.0 - col.rgb) * random(v_position.xy) * 0.1;
        
        // Contrast + Brightness
        col = pow(col, vec4(1.8)) * 0.8;
        
        // Color curves
        col.rgb = col.rgb * vec3(1.2, 1.1, 1.0) + vec3(0.0, 0.05, 0.2);
        
        // Ambient Occlusion
        //col = ambientOcclusion(col, depth, v_position.xy);                
        
        // Invert
        //col.rgb = 1.0 - col.rgb;
        
        // Fog
        //col.rgb = col.rgb + vec3((depth - 0.992) * 200.0);         
                        
        outColor = col;
    }
`;

// language=GLSL
let postVertexShader = `
    #version 300 es
    
    layout(location=0) in vec4 position;
    out vec4 v_position;
    
    void main() {
        v_position = position;
        gl_Position = position * 2.0 - 1.0;
    }
`;

async function loadTexture(fileName) {
    return await createImageBitmap(await (await fetch("images/" + fileName)).blob());
}

(async () => {

    let lightColors = [vec3.fromValues(1.0, 0.5, 0.7), vec3.fromValues(0.5, 0.6, 1.0)];

    app.clearColor(0.97, 0.21, 0.54, 1.0)
        .enable(PicoGL.DEPTH_TEST)
        .enable(PicoGL.CULL_FACE);

    let program = app.createProgram(vertexShader.trim(), fragmentShader.trim());
    let postProgram = app.createProgram(postVertexShader.trim(), postFragmentShader.trim());

    let vertexArray = app.createVertexArray()
        .vertexAttributeBuffer(0, app.createVertexBuffer(PicoGL.FLOAT, 3, positions))
        .vertexAttributeBuffer(1, app.createVertexBuffer(PicoGL.FLOAT, 3, normals))
        .vertexAttributeBuffer(2, app.createVertexBuffer(PicoGL.FLOAT, 2, uvs))
        .indexBuffer(app.createIndexBuffer(PicoGL.UNSIGNED_INT, 3, indices));

    let postArray = app.createVertexArray()
        .vertexAttributeBuffer(0, app.createVertexBuffer(PicoGL.FLOAT, 2, postPositions))
        .indexBuffer(app.createIndexBuffer(PicoGL.UNSIGNED_INT, 3, postIndices));

    let colorTarget = app.createTexture2D(app.width, app.height, {magFilter: PicoGL.LINEAR, wrapS: PicoGL.CLAMP_TO_EDGE, wrapR: PicoGL.CLAMP_TO_EDGE});
    let depthTarget = app.createTexture2D(app.width, app.height, {internalFormat: PicoGL.DEPTH_COMPONENT32F, type: PicoGL.FLOAT});
    let buffer = app.createFramebuffer().colorTarget(0, colorTarget).depthTarget(depthTarget);

    let projectionMatrix = mat4.create();
    let viewMatrix = mat4.create();
    let viewProjectionMatrix = mat4.create();
    let modelMatrix = mat4.create();

    let drawCall = app.createDrawCall(program, vertexArray)
        .uniform("ambientLightColor", ambientLightColor);

    let postDrawCall = app.createDrawCall(postProgram, postArray)
        .texture("tex", colorTarget)
        .texture("depthTex", depthTarget)
        .texture("noiseTex", app.createTexture2D(await loadTexture("noise.png")));

    const tex = await loadTexture("nautilus.jpg");
    drawCall.texture("tex", app.createTexture2D(tex, tex.width, tex.height, {
        magFilter: PicoGL.LINEAR,
        minFilter: PicoGL.LINEAR_MIPMAP_LINEAR,
        maxAnisotropy: 10
    }));

    const cubemap = app.createCubemap({
        negX: await loadTexture("stormydays_bk.png"),
        posX: await loadTexture("stormydays_ft.png"),
        negY: await loadTexture("stormydays_dn.png"),
        posY: await loadTexture("stormydays_up.png"),
        negZ: await loadTexture("stormydays_lf.png"),
        posZ: await loadTexture("stormydays_rt.png")
    });
    drawCall.texture("cubemap", cubemap);

    let startTime = new Date().getTime() / 1000;

    let cameraPosition = vec3.fromValues(0, 0, 5);

    const positionsBuffer = new Float32Array(numberOfLights * 3);
    const colorsBuffer = new Float32Array(numberOfLights * 3);

    function draw() {
        let time = new Date().getTime() / 1000 - startTime;

        mat4.fromRotationTranslation(modelMatrix, quat.fromEuler(quat.create(), -90, time * 20, 0), vec3.fromValues(0, -0.2, 0));

        mat4.perspective(projectionMatrix, Math.PI / 6, app.width / app.height, 0.1, 100.0);
        mat4.lookAt(viewMatrix, cameraPosition, vec3.fromValues(0, 0, 0), vec3.fromValues(0, 1, 0));
        mat4.multiply(viewProjectionMatrix, projectionMatrix, viewMatrix);

        app.drawFramebuffer(buffer);
        app.viewport(0, 0, colorTarget.width, colorTarget.height);

        app.enable(PicoGL.DEPTH_TEST)
            .enable(PicoGL.CULL_FACE)
            .clear();

        drawCall.uniform("viewProjectionMatrix", viewProjectionMatrix);
        drawCall.uniform("modelMatrix", modelMatrix);
        drawCall.uniform("cameraPosition", cameraPosition);

        for (let i = 0; i < numberOfLights; i++) {
            vec3.rotateZ(lightPositions[i], lightInitialPositions[i], vec3.fromValues(0, 0, 0), time * 4);
            positionsBuffer.set(lightPositions[i], i * 3);
            colorsBuffer.set(lightColors[i], i * 3);
        }

        drawCall.uniform("lightPositions[0]", positionsBuffer);
        drawCall.uniform("lightColors[0]", colorsBuffer);

        drawCall.draw();

        app.defaultDrawFramebuffer();
        app.viewport(0, 0, app.width, app.height);

        app.disable(PicoGL.DEPTH_TEST)
            .disable(PicoGL.CULL_FACE);
        postDrawCall.uniform("time", time);
        postDrawCall.draw();

        requestAnimationFrame(draw);
    }
    requestAnimationFrame(draw);
})();
