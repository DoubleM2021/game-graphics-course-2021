import PicoGL from "../node_modules/picogl/build/module/picogl.js";
import {mat4, vec3} from "../node_modules/gl-matrix/esm/index.js";

import {positions, normals, indices} from "../blender/magnetto.js"

//The indices for the post-processing
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


let ambientLightColor = vec3.fromValues(0.5, 0.05, 0.2);
let numberOfLights = 3;
let lightColors = [vec3.fromValues(1.0, 0.0, 0.2), vec3.fromValues(0.0, 0.1, 0.2), vec3.fromValues(-3.0, 0.5, 0.2)];
let lightInitialPositions = [vec3.fromValues(5, 0, 2), vec3.fromValues(-5, 0, 2), vec3.fromValues(1, 1, 1)];
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
    
    in vec3 vPosition;    
    in vec3 vNormal;
    in vec4 vColor;    
    
    out vec4 outColor;        
    
    void main() {                      
        // For Phong shading (per-fragment) move color calculation from vertex to fragment shader
        outColor = calculateLights(normalize(vNormal), vPosition);
        //outColor = vColor;
    }
`;

// language=GLSL
let vertexShader = `
    #version 300 es
    ${lightCalculationShader}
        
    layout(location=0) in vec4 position;
    layout(location=1) in vec4 normal;
    
    uniform mat4 viewProjectionMatrix;
    uniform mat4 modelMatrix;            
    
    out vec3 vPosition;    
    out vec3 vNormal;
    out vec4 vColor;
    
    void main() {
        vec4 worldPosition = modelMatrix * position;
        
        vPosition = worldPosition.xyz;        
        vNormal = (modelMatrix * normal).xyz;
        
        // For Gouraud shading (per-vertex) move color calculation from fragment to vertex shader
        //vColor = calculateLights(normalize(vNormal), vPosition);
        
        gl_Position = viewProjectionMatrix * worldPosition;                        
    }
`;


app.enable(PicoGL.DEPTH_TEST)
   .enable(PicoGL.CULL_FACE);

let program = app.createProgram(vertexShader.trim(), fragmentShader.trim());

let vertexArray = app.createVertexArray()
    .vertexAttributeBuffer(0, app.createVertexBuffer(PicoGL.FLOAT, 3, positions))
    .vertexAttributeBuffer(1, app.createVertexBuffer(PicoGL.FLOAT, 3, normals))
    .indexBuffer(app.createIndexBuffer(PicoGL.UNSIGNED_INT, 3, indices));

let projectionMatrix = mat4.create();
let viewMatrix = mat4.create();
let viewProjectionMatrix = mat4.create();
let modelMatrix = mat4.create();

let drawCall = app.createDrawCall(program, vertexArray)
    .uniform("ambientLightColor", ambientLightColor);

let startTime = new Date().getTime() / 1000;

let cameraPosition = vec3.fromValues(0, 20, 5);
mat4.fromXRotation(modelMatrix, -Math.PI / 2);

const positionsBuffer = new Float32Array(numberOfLights * 3);
const colorsBuffer = new Float32Array(numberOfLights * 3);

function draw() {
    let time = new Date().getTime() / 500 - startTime;

    mat4.perspective(projectionMatrix, Math.PI / 4, app.width / app.height, 0.1, 50.0);
    mat4.lookAt(viewMatrix, cameraPosition, vec3.fromValues(0, 4, 0), vec3.fromValues(0, 1, 0));
    mat4.multiply(viewProjectionMatrix, projectionMatrix, viewMatrix);

    drawCall.uniform("viewProjectionMatrix", viewProjectionMatrix);
    drawCall.uniform("modelMatrix", modelMatrix);
    drawCall.uniform("cameraPosition", cameraPosition);

    for (let i = 0; i < numberOfLights; i++) {
        vec3.rotateZ(lightPositions[i], lightInitialPositions[i], vec3.fromValues(1, 7, 4), time);
        positionsBuffer.set(lightPositions[i], i * 3);
        colorsBuffer.set(lightColors[i], i * 3);
    }

    drawCall.uniform("lightPositions[0]", positionsBuffer);
    drawCall.uniform("lightColors[0]", colorsBuffer);

    app.clear();
    drawCall.draw();

    requestAnimationFrame(draw);
}
requestAnimationFrame(draw);
