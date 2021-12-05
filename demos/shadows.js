// Shadow mapping demo

import PicoGL from "../node_modules/picogl/build/module/picogl.js";
import {mat4, vec3, vec4, quat} from "../node_modules/gl-matrix/esm/index.js";

import {positions, normals, indices} from "../blender/cranium.js"

// language=GLSL
let fragmentShader = `
    #version 300 es    
    precision highp float;    
    precision highp sampler2DShadow;
    
    uniform vec4 baseColor;
    uniform vec4 ambientColor;
    uniform vec3 lightPosition;
    uniform vec3 cameraPosition;    
    uniform sampler2DShadow shadowMap;
    
    in vec3 vPosition;
    in vec3 vNormal;
    in vec4 vPositionFromLight;
    in vec3 vModelPosition;
    out vec4 fragColor;
    
    void main() {
        vec3 shadowCoord = (vPositionFromLight.xyz / vPositionFromLight.w) / 5.0 + 3.5;        
        float shadow = texture(shadowMap, shadowCoord);
        
        vec3 normal = normalize(vNormal);
        vec3 eyeDirection = normalize(cameraPosition - vPosition);
        vec3 lightDirection = normalize(lightPosition - vPosition);        
        vec3 reflectionDirection = reflect(-lightDirection, normal);
        
        float diffuse = max(dot(lightDirection, normal), 0.0) * max(shadow, 0.2);        
        float specular = shadow * pow(max(dot(reflectionDirection, eyeDirection), 0.0), 100.0) * 0.7;
        fragColor = vec4(diffuse * baseColor.rgb + ambientColor.rgb + specular, baseColor.a);
    }
`;

// language=GLSL
let vertexShader = `
    #version 300 es
    
    layout(location=0) in vec4 position;
    layout(location=1) in vec3 normal;
    
    uniform mat4 modelMatrix;
    uniform mat4 modelViewProjectionMatrix;
    uniform mat4 lightModelViewProjectionMatrix;
    
    out vec3 vPosition;
    out vec3 vNormal;
    out vec4 vPositionFromLight;
    out vec3 vModelPosition;
    
    void main() {
        gl_Position = modelViewProjectionMatrix * position;
        vModelPosition = vec3(position);
        vPosition = vec3(modelMatrix * position);
        vNormal = vec3(modelMatrix * vec4(normal, 0.0));
        vPositionFromLight = lightModelViewProjectionMatrix * position;
    }
`;

// language=GLSL
let shadowFragmentShader = `
    #version 300 es
    precision highp float;
    
    out vec4 fragColor;
    
    void main() {
        // Uncomment to see the depth buffer of the shadow map    
        //fragColor = vec4((gl_FragCoord.z - 0.98) * 50.0);    
    }
`;

// language=GLSL
let shadowVertexShader = `
    #version 300 es
    layout(location=0) in vec4 position;
    uniform mat4 lightModelViewProjectionMatrix;
    
    void main() {
        gl_Position = lightModelViewProjectionMatrix * position;
    }
`;

let bgColor = vec4.fromValues(1.0, 0.2, 0.3, 1.0);
let fgColor = vec4.fromValues(1.0, 0.9, 0.5, 1.0);

app.enable(PicoGL.DEPTH_TEST)
   .enable(PicoGL.CULL_FACE)
   .clearColor(bgColor[0], bgColor[1], bgColor[2], bgColor[3]);

let program = app.createProgram(vertexShader, fragmentShader);
let shadowProgram = app.createProgram(shadowVertexShader, shadowFragmentShader);

let vertexArray = app.createVertexArray()
    .vertexAttributeBuffer(0, app.createVertexBuffer(PicoGL.FLOAT, 3, positions))
    .vertexAttributeBuffer(1, app.createVertexBuffer(PicoGL.FLOAT, 3, normals))
    .indexBuffer(app.createIndexBuffer(PicoGL.UNSIGNED_INT, 3, indices));

// Change the shadow texture resolution to checkout the difference
let shadowDepthTarget = app.createTexture2D(512, 512, {
    internalFormat: PicoGL.DEPTH_COMPONENT16,
    compareMode: PicoGL.COMPARE_REF_TO_TEXTURE,
    magFilter: PicoGL.LINEAR,
});
let shadowBuffer = app.createFramebuffer().depthTarget(shadowDepthTarget);

let time = 0;
let projMatrix = mat4.create();
let viewMatrix = mat4.create();
let viewProjMatrix = mat4.create();
let modelMatrix = mat4.create();
let modelViewProjectionMatrix = mat4.create();
let rotation = quat.create();
let lightModelViewProjectionMatrix = mat4.create();

let cameraPosition = vec3.fromValues(0, 2, 4);
let lightPosition = vec3.fromValues(5, 5, 2.5);
let lightViewMatrix = mat4.create();
let lightViewProjMatrix = mat4.create();
mat4.lookAt(lightViewMatrix, lightPosition, vec3.fromValues(0, 0, 0), vec3.fromValues(0, 1, 0));


let drawCall = app.createDrawCall(program, vertexArray)
    .uniform("baseColor", fgColor)
    .uniform("ambientColor", vec4.scale(vec4.create(), bgColor, 0.7))
    .uniform("modelMatrix", modelMatrix)
    .uniform("modelViewProjectionMatrix", modelViewProjectionMatrix)
    .uniform("cameraPosition", cameraPosition)
    .uniform("lightPosition", lightPosition)
    .uniform("lightModelViewProjectionMatrix", lightModelViewProjectionMatrix)
    .texture("shadowMap", shadowDepthTarget);

let shadowDrawCall = app.createDrawCall(shadowProgram, vertexArray)
    .uniform("lightModelViewProjectionMatrix", lightModelViewProjectionMatrix);

function renderShadowMap() {
    app.drawFramebuffer(shadowBuffer);
    app.viewport(0, 0, shadowDepthTarget.width, shadowDepthTarget.height);
    app.gl.cullFace(app.gl.FRONT);

    // Change the projection and view matrices to render objects from the point view of light source
    mat4.perspective(projMatrix, Math.PI * 0.1, shadowDepthTarget.width / shadowDepthTarget.height, 0.1, 100.0);
    mat4.multiply(lightViewProjMatrix, projMatrix, lightViewMatrix);

    drawObjects(shadowDrawCall);

    app.gl.cullFace(app.gl.BACK);
    app.defaultDrawFramebuffer();
    app.defaultViewport();
}

function drawObjects(dc) {
    app.clear();

    // Middle object
    quat.fromEuler(rotation, time * 48.24, time * 56.97, 0);
    mat4.fromRotationTranslationScale(modelMatrix, rotation, vec3.fromValues(0, 0, 0), [0.8, 0.8, 0.8]);
    mat4.multiply(modelViewProjectionMatrix, viewProjMatrix, modelMatrix);
    mat4.multiply(lightModelViewProjectionMatrix, lightViewProjMatrix, modelMatrix);

    dc.draw();

    // Large object
    quat.fromEuler(rotation, time * 12, time * 14, 0);
    mat4.fromRotationTranslationScale(modelMatrix, rotation, vec3.fromValues(-2.4, -2.4, -1.2), [2, 2, 2]);
    mat4.multiply(modelViewProjectionMatrix, viewProjMatrix, modelMatrix);
    mat4.multiply(lightModelViewProjectionMatrix, lightViewProjMatrix, modelMatrix);

    dc.draw();

    // Small object
    quat.fromEuler(rotation, time * 15, time * 17, 0);
    mat4.fromRotationTranslationScale(modelMatrix, rotation, vec3.fromValues(0.9, 0.9, 0.6), [0.22, 0.22, 0.22]);
    mat4.multiply(modelViewProjectionMatrix, viewProjMatrix, modelMatrix);
    mat4.multiply(lightModelViewProjectionMatrix, lightViewProjMatrix, modelMatrix);

    dc.draw();
}

function draw() {
    time = new Date().getTime() * 0.001;

    mat4.perspective(projMatrix, Math.PI / 2.5, app.width / app.height, 0.1, 100.0);
    mat4.lookAt(viewMatrix, cameraPosition, vec3.fromValues(0, -0.5, 0), vec3.fromValues(0, 1, 0));
    mat4.multiply(viewProjMatrix, projMatrix, viewMatrix);

    renderShadowMap();
    drawObjects(drawCall);

    requestAnimationFrame(draw);
}
requestAnimationFrame(draw);
