struct Uniforms {
    mvp: mat4x4<f32>,
    rot: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec4<f32>,
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
    let n_len2 = dot(in.normal, in.normal);
    let n = select(vec3<f32>(0.0, 0.0, 1.0), normalize(in.normal), n_len2 > 1e-10);
    let key_light = normalize(vec3<f32>(0.36, 0.56, 0.74));
    let fill_light = normalize(vec3<f32>(-0.52, -0.28, 0.48));
    let d_key = abs(dot(n, key_light));
    let d_fill = abs(dot(n, fill_light));
    let half_vec = normalize(key_light + vec3<f32>(0.0, 0.0, 1.0));
    let spec = pow(abs(dot(n, half_vec)), 20.0) * 0.12;
    let shade = 0.24 + 0.62 * d_key + 0.14 * d_fill;
    let rgb = clamp(in.color.rgb * shade + vec3<f32>(spec), vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(rgb, in.color.a);
}

@fragment
fn fs_wire(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
