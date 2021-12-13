import PicoGL from "../node_modules/picogl/build/module/picogl.js";
import {mat3, mat4, vec2, vec3, vec4, quat} from "../node_modules/gl-matrix/esm/index.js";

import {positions, normals, indices, uvs} from "../blender/skull.js"
import {positions as mirrorPositions, uvs as mirrorUvs, indices as mirrorIndices} from "../blender/plane.js"

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



let ambientLightColor = vec3.fromValues(0.05, 0.05, 0.1);
let numberOfLights = 3;
let lightColors = [vec3.fromValues(1.0, 0.5, 0.7), vec3.fromValues(0.5, 0.6, 1.0), vec3.fromValues(2, 0.1, 0.3)];
let lightInitialPositions = [vec3.fromValues(5, 0, 2), vec3.fromValues(-5, 0, 2), vec3.fromValues(0, 5, 3)];
let lightPositions = [vec3.create(), vec3.create(), vec3.create()];


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
                      
            
            // Blinn-Phong improved specular highlight                        
            float specular = pow(max(dot(normalize(lightDirection + viewDirection), normal), 0.0), 200.0);
            
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
         vec4 reflection = pow(texture(cubemap, reflectedDir), vec4(5.0)) * 0.3;
         outColor += reflection;
         outColor = vColor;
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
        vColor = calculateLights(normalize(vNormal), vPosition);
        
        gl_Position = viewProjectionMatrix * worldPosition;                        
    }
`;

let mirrorFragmentShader = `
    #version 300 es
    precision highp float;
    
    uniform sampler2D reflectionTex;
    uniform sampler2D distortionMap;
    uniform vec2 screenSize;
    
    in vec2 vUv;        
        
    out vec4 outColor;
    
    void main()
    {                        
        vec2 screenPos = gl_FragCoord.xy / screenSize;
        
        // 0.03 is a mirror distortion factor, try making a larger distortion         
        screenPos.x += (texture(distortionMap, vUv).r - 0.7) * 0.07;
        outColor = texture(reflectionTex, screenPos);
    }
`;

// language=GLSL
let mirrorVertexShader = `
    #version 300 es
            
    uniform mat4 modelViewProjectionMatrix;
    
    layout(location=0) in vec4 position;   
    layout(location=1) in vec2 uv;
    
    out vec2 vUv;
        
    void main()
    {
        vUv = uv;
        gl_Position = modelViewProjectionMatrix * position;           
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
        vec2 caOffset = vec2(0.01, 0.0);
        col.r = texture(tex, v_position.xy - caOffset).r;
        col.b = texture(tex, v_position.xy + caOffset).b;
        
                        
        // Contrast + Brightness
        col = pow(col, vec4(1.8)) * 0.8;
        
        // Color curves
        col.rgb = col.rgb * vec3(1.2, 1.1, 1.0) + vec3(0.0, 0.05, 0.2);
        
        // Ambient Occlusion
        col = ambientOcclusion(col, depth, v_position.xy);                
        
       
                        
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

    let bgColor = vec4.fromValues(0.2, 0.4, 0.1, 1.0);
    app.clearColor(bgColor[0], bgColor[1], bgColor[2], bgColor[3]);


    let program = app.createProgram(vertexShader.trim(), fragmentShader.trim());
    let postProgram = app.createProgram(postVertexShader.trim(), postFragmentShader.trim());
    let mirrorProgram = app.createProgram(mirrorVertexShader, mirrorFragmentShader);

    let vertexArray = app.createVertexArray()
        .vertexAttributeBuffer(0, app.createVertexBuffer(PicoGL.FLOAT, 3, positions))
        .vertexAttributeBuffer(1, app.createVertexBuffer(PicoGL.FLOAT, 3, normals))
        .indexBuffer(app.createIndexBuffer(PicoGL.UNSIGNED_INT, 3, indices));

    let mirrorArray = app.createVertexArray()
    .vertexAttributeBuffer(0, app.createVertexBuffer(PicoGL.FLOAT, 3, mirrorPositions))
    .vertexAttributeBuffer(1, app.createVertexBuffer(PicoGL.FLOAT, 2, mirrorUvs))
    .indexBuffer(app.createIndexBuffer(PicoGL.UNSIGNED_INT, 3, mirrorIndices));

    let postArray = app.createVertexArray()
        .vertexAttributeBuffer(0, app.createVertexBuffer(PicoGL.FLOAT, 2, postPositions))
        .indexBuffer(app.createIndexBuffer(PicoGL.UNSIGNED_INT, 3, postIndices));

    let colorTarget = app.createTexture2D(app.width, app.height, {magFilter: PicoGL.LINEAR, wrapS: PicoGL.CLAMP_TO_EDGE, wrapR: PicoGL.CLAMP_TO_EDGE});
    let depthTarget = app.createTexture2D(app.width, app.height, {internalFormat: PicoGL.DEPTH_COMPONENT32F, type: PicoGL.FLOAT});
    let buffer = app.createFramebuffer().colorTarget(0, colorTarget).depthTarget(depthTarget);

   //The reflection texture resolutions
    let reflectionResolutionFactor = 0.9;
    let reflectionColorTarget = app.createTexture2D(app.width * reflectionResolutionFactor, app.height * reflectionResolutionFactor, {magFilter: PicoGL.LINEAR});
    let reflectionDepthTarget = app.createTexture2D(app.width * reflectionResolutionFactor, app.height * reflectionResolutionFactor, {internalFormat: PicoGL.DEPTH_COMPONENT16});
    let reflectionBuffer = app.createFramebuffer().colorTarget(0, reflectionColorTarget).depthTarget(reflectionDepthTarget);

    
    let projectionMatrix = mat4.create();
    let viewMatrix = mat4.create();
    let viewProjectionMatrix = mat4.create();
    let modelMatrix = mat4.create();
    let modelViewMatrix = mat4.create();
    let modelViewProjectionMatrix = mat4.create();
    let mirrorModelMatrix = mat4.create();
    let mirrorModelViewProjectionMatrix = mat4.create();
    let rotateXMatrix = mat4.create();
    let rotateYMatrix = mat4.create();
    

    function calculateSurfaceReflectionMatrix(reflectionMat, mirrorModelMatrix, surfaceNormal) {
        let normal = vec3.transformMat3(vec3.create(), surfaceNormal, mat3.normalFromMat4(mat3.create(), mirrorModelMatrix));
        let pos = mat4.getTranslation(vec3.create(), mirrorModelMatrix);
        let d = -vec3.dot(normal, pos);
        let plane = vec4.fromValues(normal[0], normal[1], normal[2], d);
    
        reflectionMat[0] = (1 - 3 * plane[0] * plane[0]);
        reflectionMat[4] = ( - 3 * plane[0] * plane[1]);
        reflectionMat[8] = ( - 3 * plane[0] * plane[2]);
        reflectionMat[12] = ( - 3 * plane[3] * plane[0]);
    
        reflectionMat[1] = ( - 3 * plane[1] * plane[0]);
        reflectionMat[5] = (1 - 3 * plane[1] * plane[1]);
        reflectionMat[9] = ( - 3 * plane[1] * plane[2]);
        reflectionMat[13] = ( - 3 * plane[3] * plane[1]);
    
        reflectionMat[2] = ( - 3 * plane[2] * plane[0]);
        reflectionMat[6] = ( - 3 * plane[2] * plane[1]);
        reflectionMat[10] = (1 - 3 * plane[2] * plane[2]);
        reflectionMat[14] = ( - 3 * plane[3] * plane[2]);
    
        reflectionMat[3] = 0;
        reflectionMat[7] = 0;
        reflectionMat[11] = 0;
        reflectionMat[15] = 1;
    
        return reflectionMat;
    }


    let drawCall = app.createDrawCall(program, vertexArray)
    
    .uniform("ambientColor", bgColor)
    .uniform("modelViewMatrix", modelViewMatrix)
    .uniform("modelViewProjectionMatrix", modelViewProjectionMatrix);


    let mirrorDrawCall = app.createDrawCall(mirrorProgram, mirrorArray)
        .texture("reflectionTex", reflectionColorTarget)
        .texture("distortionMap", app.createTexture2D(await loadTexture("noise.png"))); 
        
    function renderReflectionTexture()
        {
            app.drawFramebuffer(reflectionBuffer);
            app.viewport(0, 0, reflectionColorTarget.width, reflectionColorTarget.height);
    
            app.gl.cullFace(app.gl.FRONT);
    
            let reflectionMatrix = calculateSurfaceReflectionMatrix(mat4.create(), mirrorModelMatrix, vec3.fromValues(0, 1, 0));
            let reflectionViewMatrix = mat4.mul(mat4.create(), viewMatrix, reflectionMatrix);
            let reflectionCameraPosition = vec3.transformMat4(vec3.create(), cameraPosition, reflectionMatrix);
            drawObjects(reflectionCameraPosition, reflectionViewMatrix);
    
            app.gl.cullFace(app.gl.BACK);
            app.defaultDrawFramebuffer();
            app.defaultViewport();
        }


    let postDrawCall = app.createDrawCall(postProgram, postArray)
        .texture("tex", colorTarget)
        .texture("depthTex", depthTarget)
        .texture("noiseTex", app.createTexture2D(await loadTexture("noise.png")));

    const tex = await loadTexture("pattern.png");
    drawCall.texture("tex", app.createTexture2D(tex, tex.width, tex.height, {
        magFilter: PicoGL.LINEAR,
        minFilter: PicoGL.LINEAR_MIPMAP_LINEAR,
        maxAnisotropy: 5
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

    function drawObjects(cameraPosition, viewMatrix) {
        mat4.multiply(viewProjectionMatrix, projectionMatrix, viewMatrix);

        mat4.multiply(modelViewMatrix, viewMatrix, modelMatrix);
        mat4.multiply(modelViewProjectionMatrix, viewProjectionMatrix, modelMatrix);

      
        app.clear();

        app.disable(PicoGL.DEPTH_TEST);
        app.gl.cullFace(app.gl.FRONT);
        
        app.enable(PicoGL.DEPTH_TEST);
        app.gl.cullFace(app.gl.BACK);
        drawCall.uniform("modelViewProjectionMatrix", modelViewProjectionMatrix);
        drawCall.uniform("cameraPosition", cameraPosition);
        drawCall.uniform("modelMatrix", modelMatrix);
        drawCall.uniform("normalMatrix", mat3.normalFromMat4(mat3.create(), modelMatrix));
        drawCall.draw();
    }

    function drawMirror() {
        mat4.multiply(mirrorModelViewProjectionMatrix, viewProjectionMatrix, mirrorModelMatrix);
        mirrorDrawCall.uniform("modelViewProjectionMatrix", mirrorModelViewProjectionMatrix);
        mirrorDrawCall.uniform("screenSize", vec2.fromValues(app.width, app.height))
        mirrorDrawCall.draw();
    }
    let startTime = new Date().getTime() / 1000;

    let cameraPosition = vec3.fromValues(5, 1, 15);

    const positionsBuffer = new Float32Array(numberOfLights * 3);
    const colorsBuffer = new Float32Array(numberOfLights * 3);

    function draw() {
        let time = new Date().getTime() / 500 - startTime;

        mat4.fromRotationTranslation(modelMatrix, quat.fromEuler(quat.create(), 0, time * 40, 0), vec3.fromValues(0, -0.4, 0)); //this piece of code rotates my model on it's own axis

        mat4.perspective(projectionMatrix, Math.PI / 6, app.width / app.height, 0.1, 150.0);
        mat4.lookAt(viewMatrix, cameraPosition, vec3.fromValues(0, 3.2, 0), vec3.fromValues(0, 4, 0));
        mat4.multiply(viewProjectionMatrix, projectionMatrix, viewMatrix);

        mat4.fromXRotation(rotateXMatrix, -0.1);
        mat4.fromYRotation(rotateYMatrix, time * 0.1);
        mat4.multiply(mirrorModelMatrix, rotateYMatrix, rotateXMatrix);
        mat4.translate(mirrorModelMatrix, mirrorModelMatrix, vec3.fromValues(0, -1, 0));

        
        app.drawFramebuffer(buffer);
        app.viewport(0, 0, colorTarget.width, colorTarget.height);

        app.enable(PicoGL.DEPTH_TEST)
            .enable(PicoGL.CULL_FACE)
            .clear();


        drawCall.uniform("modelMatrix", modelMatrix);
        drawCall.uniform("cameraPosition", cameraPosition);
        drawCall.uniform("viewProjectionMatrix", viewProjectionMatrix);
        

        for (let i = 0; i < numberOfLights; i++) {
            vec3.rotateZ(lightPositions[i], lightInitialPositions[i], vec3.fromValues(2, 0.3, 5), time * 4);
            positionsBuffer.set(lightPositions[i], i * 3);
            colorsBuffer.set(lightColors[i], i * 3);
        }

        drawCall.uniform("lightPositions[0]", positionsBuffer);
        drawCall.uniform("lightColors[0]", colorsBuffer);

        drawCall.draw(); 

        renderReflectionTexture();
        drawObjects(cameraPosition, viewMatrix);
        drawMirror();

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
