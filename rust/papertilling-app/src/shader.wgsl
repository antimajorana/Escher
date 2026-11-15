struct Uniforms {
    mvp: mat4x4<f32>,
    rot: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = uniforms.mvp * vec4<f32>(in.position, 1.0);
    out.normal = (uniforms.rot * vec4<f32>(in.normal, 0.0)).xyz;
    out.color = in.color;
    return out;
}

@fragment
fn fs_face(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let light = normalize(vec3<f32>(0.38, 0.58, 0.72));
    let two_sided = abs(dot(n, light));
    let shade = 0.24 + 0.76 * two_sided;
    return vec4<f32>(in.color * shade, 1.0);
}

@fragment
fn fs_wire(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
