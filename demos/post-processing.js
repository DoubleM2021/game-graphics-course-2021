import PicoGL from "../node_modules/picogl/build/module/picogl.js";
import {mat4, vec3, vec4, quat} from "../node_modules/gl-matrix/esm/index.js";

import {positions, normals, indices} from "../blender/monkey.js"

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


// language=GLSL
let fragmentShader = `
    #version 300 es
    precision highp float;

    uniform vec4 ambientColor;
    uniform vec4 diffuseColor;
    
    in vec3 vViewNormal;
    
    out vec4 outColor;       
    
    void main() {                      
        outColor =  + pow(vViewNormal.y, 5.0) + diffuseColor * abs(vViewNormal.y) + ambientColor;
    }
`;

// language=GLSL
let vertexShader = `
    #version 300 es
    
    uniform mat4 modelViewMatrix;
    uniform mat4 modelViewProjectionMatrix;
    
    layout(location=0) in vec3 position;
    layout(location=1) in vec3 normal;
    
    out vec3 vViewNormal;
    
    void main()
    {
        gl_Position = modelViewProjectionMatrix * vec4(position, 1.0);
        vViewNormal = (modelViewMatrix * vec4(normalize(normal), 0.0)).xyz;
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
    let bgColor = vec4.fromValues(0.1, 0.1, 0.1, 1.0);
    app.clearColor(bgColor[0], bgColor[1], bgColor[2], bgColor[3]);

    let program = app.createProgram(vertexShader.trim(), fragmentShader.trim());
    let postProgram = app.createProgram(postVertexShader.trim(), postFragmentShader.trim());

    let vertexArray = app.createVertexArray()
        .vertexAttributeBuffer(0, app.createVertexBuffer(PicoGL.FLOAT, 3, positions))
        .vertexAttributeBuffer(1, app.createVertexBuffer(PicoGL.FLOAT, 3, normals))
        .indexBuffer(app.createIndexBuffer(PicoGL.UNSIGNED_INT, 3, indices));

    let postArray = app.createVertexArray()
        .vertexAttributeBuffer(0, app.createVertexBuffer(PicoGL.FLOAT, 2, postPositions))
        .indexBuffer(app.createIndexBuffer(PicoGL.UNSIGNED_INT, 3, postIndices));

    let colorTarget = app.createTexture2D(app.width, app.height, {magFilter: PicoGL.LINEAR, wrapS: PicoGL.CLAMP_TO_EDGE, wrapR: PicoGL.CLAMP_TO_EDGE});
    let depthTarget = app.createTexture2D(app.width, app.height, {internalFormat: PicoGL.DEPTH_COMPONENT32F, type: PicoGL.FLOAT});
    let buffer = app.createFramebuffer().colorTarget(0, colorTarget).depthTarget(depthTarget);

    let projectionMatrix = mat4.create();
    let viewMatrix = mat4.create();
    let viewProjMatrix = mat4.create();
    let modelViewMatrix = mat4.create();
    let modelViewProjectionMatrix = mat4.create();
    let modelMatrix = mat4.create();
    let modelRotation = quat.create();

    let drawCall = app.createDrawCall(program, vertexArray)
        .uniform("ambientColor", bgColor)
        .uniform("modelViewMatrix", modelViewMatrix)
        .uniform("modelViewProjectionMatrix", modelViewProjectionMatrix);

    let postDrawCall = app.createDrawCall(postProgram, postArray)
        .texture("tex", colorTarget)
        .texture("depthTex", depthTarget)
        .texture("noiseTex", app.createTexture2D(await loadTexture("noise.png")));

    let time = 0, previousTime = 0, rotation = 0;

    function draw(timestamp) {
        requestAnimationFrame(draw);
        previousTime = time;
        time = timestamp / 1000;
        if (!document.hasFocus()) return;

        let cameraPosition = vec3.fromValues(0, 0, 9);
        rotation += (time - previousTime) * 0.5;
        mat4.perspective(projectionMatrix, Math.PI / 10, app.width / app.height, 0.05, 50.0);
        mat4.lookAt(viewMatrix, cameraPosition, vec3.fromValues(0, 0, 0), vec3.fromValues(0, 1, 0));
        quat.fromEuler(modelRotation, Math.cos(rotation) * 20 - 90, Math.sin(rotation) * 20, 0)
        mat4.multiply(viewProjMatrix, projectionMatrix, viewMatrix);

        mat4.multiply(modelViewMatrix, viewMatrix, modelMatrix);
        mat4.multiply(modelViewProjectionMatrix, viewProjMatrix, modelMatrix);

        app.drawFramebuffer(buffer);
        app.viewport(0, 0, colorTarget.width, colorTarget.height);

        app.enable(PicoGL.DEPTH_TEST)
           .enable(PicoGL.CULL_FACE)
           .clear();

        drawCall.uniform("diffuseColor", vec4.fromValues(0.3, 0.0, 1.0, 1.0));
        mat4.fromRotationTranslation(modelMatrix, modelRotation, vec3.fromValues(-1.5, 0, -2));
        mat4.multiply(modelViewMatrix, viewMatrix, modelMatrix);
        mat4.multiply(modelViewProjectionMatrix, viewProjMatrix, modelMatrix);
        drawCall.draw();

        drawCall.uniform("diffuseColor", vec4.fromValues(0.1, 1.0, 0.2, 1.0));
        mat4.fromRotationTranslation(modelMatrix, modelRotation, vec3.fromValues(0, 0, 0));
        mat4.multiply(modelViewMatrix, viewMatrix, modelMatrix);
        mat4.multiply(modelViewProjectionMatrix, viewProjMatrix, modelMatrix);
        drawCall.draw();

        drawCall.uniform("diffuseColor", vec4.fromValues(1.0, 0.0, 0.2, 1.0));
        mat4.fromRotationTranslation(modelMatrix, modelRotation, vec3.fromValues(1.5, 0, 2));
        mat4.multiply(modelViewMatrix, viewMatrix, modelMatrix);
        mat4.multiply(modelViewProjectionMatrix, viewProjMatrix, modelMatrix);
        drawCall.draw();

        app.defaultDrawFramebuffer();
        app.viewport(0, 0, app.width, app.height);

        app.disable(PicoGL.DEPTH_TEST)
           .disable(PicoGL.CULL_FACE);
        postDrawCall.uniform("time", time);
        postDrawCall.draw();
    }
    requestAnimationFrame(draw);
})();
