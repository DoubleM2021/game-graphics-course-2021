// *********************************************************************************************************************
// **                                                                                                                 **
// **             Texturing example, Cube is mapped with 2D texture, skybox is mapped with a Cubemap                  **
// **                                                                                                                 **
// *********************************************************************************************************************

// * Change textures
// * Combine several textures in fragment shaders
// * Distort UV coordinates
// * Change texture filtering for pixel graphics
// * Use wrapping modes for texture tiling

import PicoGL from "../node_modules/picogl/build/module/picogl.js";
import {mat4, vec3} from "../node_modules/gl-matrix/esm/index.js";

import {positions, normals, uvs, indices} from "../blender/cranium.js"

const skyboxPositions = new Float32Array([
    -1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    -1.0, -1.0, 1.0,
    1.0, -1.0, 1.0
]);

const skyboxIndices = new Uint32Array([
    0, 2, 1,
    2, 3, 1
]);


// language=GLSL
let fragmentShader = `
    #version 300 es
    precision highp float;
    
    uniform sampler2D tex;    
    
    in vec2 v_uv;
    
    out vec4 outColor;
    
    void main()
    {        
        outColor = texture(tex, v_uv);
    }
`;

// language=GLSL
let vertexShader = `
    #version 300 es
            
    uniform mat4 modelViewProjectionMatrix;
    
    layout(location=0) in vec3 position;
    layout(location=1) in vec3 normal;
    layout(location=2) in vec2 uv;
        
    out vec2 v_uv;
    
    void main()
    {
        gl_Position = modelViewProjectionMatrix * vec4(position, 1.5);           
        v_uv = uv;
    }
`;


// language=GLSL
let skyboxFragmentShader = `
    #version 300 es
    precision mediump float;
    
    uniform samplerCube cubemap;
    uniform mat4 viewProjectionInverse;
    in vec4 v_position;
    
    out vec4 outColor;
    
    void main() {
      vec4 t = viewProjectionInverse * v_position;
      outColor = texture(cubemap, normalize(t.xyz / t.w))* vec4(.5, .3, .2, 1);
    }
`;

// language=GLSL
let skyboxVertexShader = `
    #version 300 es
    
    layout(location=0) in vec4 position;
    out vec4 v_position;
    
    void main() {
      v_position = position;
      gl_Position = position;
    }
`;

app.enable(PicoGL.CULL_FACE);

let program = app.createProgram(vertexShader.trim(), fragmentShader.trim());
let skyboxProgram = app.createProgram(skyboxVertexShader.trim(), skyboxFragmentShader.trim());

let vertexArray = app.createVertexArray()
    .vertexAttributeBuffer(0, app.createVertexBuffer(PicoGL.FLOAT, 3, positions))
    .vertexAttributeBuffer(1, app.createVertexBuffer(PicoGL.FLOAT, 3, normals))
    .vertexAttributeBuffer(2, app.createVertexBuffer(PicoGL.FLOAT, 2, uvs))
    .indexBuffer(app.createIndexBuffer(PicoGL.UNSIGNED_INT, 3, indices));

let skyboxArray = app.createVertexArray()
    .vertexAttributeBuffer(0, app.createVertexBuffer(PicoGL.FLOAT, 3, skyboxPositions))
    .indexBuffer(app.createIndexBuffer(PicoGL.UNSIGNED_INT, 3, skyboxIndices));

let projMatrix = mat4.create();
let viewMatrix = mat4.create();
let viewProjMatrix = mat4.create();
let modelMatrix = mat4.create();
let modelViewMatrix = mat4.create();
let modelViewProjectionMatrix = mat4.create();
let rotateXMatrix = mat4.create();
let rotateYMatrix = mat4.create();
let skyboxViewProjectionInverse = mat4.create();

async function loadTexture(fileName) {
    return await createImageBitmap(await (await fetch("images/" + fileName)).blob());
}

(async () => {
    const tex = await loadTexture("checkerboard.png");
    let drawCall = app.createDrawCall(program, vertexArray)
        .texture("tex", app.createTexture2D(tex, tex.width, tex.height, {
            magFilter: PicoGL.LINEAR,
            minFilter: PicoGL.LINEAR_MIPMAP_LINEAR,
            maxAnisotropy: 10,
            wrapS: PicoGL.REPEAT,
            wrapT: PicoGL.REPEAT
        }));

    let skyboxDrawCall = app.createDrawCall(skyboxProgram, skyboxArray)
        .texture("cubemap", app.createCubemap({
            negX: await loadTexture("stormydays_bk.png"),
            posX: await loadTexture("stormydays_ft.png"),
            negY: await loadTexture("stormydays_dn.png"),
            posY: await loadTexture("stormydays_up.png"),
            negZ: await loadTexture("stormydays_lf.png"),
            posZ: await loadTexture("stormydays_rt.png")
        }));

    let startTime = new Date().getTime() / 1000;


    function draw() {
        let time = new Date().getTime() / 1000 - startTime;

        mat4.perspective(projMatrix, Math.PI / 2, app.width / app.height, 0.1, 100.0);
        let camPos = vec3.rotateY(vec3.create(), vec3.fromValues(0, 0.5, 2), vec3.fromValues(0, 0, 0), time * 0.05);
        mat4.lookAt(viewMatrix, camPos, vec3.fromValues(0, 0, 0), vec3.fromValues(0, 1, 0));
        mat4.multiply(viewProjMatrix, projMatrix, viewMatrix);

        mat4.fromXRotation(rotateXMatrix, time * 0.1136);
        mat4.fromZRotation(rotateYMatrix, time * 0.2235);
        mat4.multiply(modelMatrix, rotateXMatrix, rotateYMatrix);

        mat4.multiply(modelViewMatrix, viewMatrix, modelMatrix);
        mat4.multiply(modelViewProjectionMatrix, viewProjMatrix, modelMatrix);

        let skyboxViewProjectionMatrix = mat4.create();
        mat4.mul(skyboxViewProjectionMatrix, projMatrix, viewMatrix);
        mat4.invert(skyboxViewProjectionInverse, skyboxViewProjectionMatrix);

        app.clear();

        app.disable(PicoGL.DEPTH_TEST);
        skyboxDrawCall.uniform("viewProjectionInverse", skyboxViewProjectionInverse);
        skyboxDrawCall.draw();

        app.enable(PicoGL.DEPTH_TEST);
        drawCall.uniform("modelViewProjectionMatrix", modelViewProjectionMatrix);
        drawCall.draw();

        requestAnimationFrame(draw);
    }

    requestAnimationFrame(draw);
})();
