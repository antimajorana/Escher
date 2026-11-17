struct Uniforms {
    mvp: mat4x4<f32>,
    rot: mat4x4<f32>,
    theme: vec4<f32>,
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
    let c = in.color.rgb;
    let max_c = max(c.r, max(c.g, c.b));
    let min_c = min(c.r, min(c.g, c.b));
    let chroma = max_c - min_c;
    let luma = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    if uniforms.theme.x > 0.5 {
        // Light Paper backdrop: map near-white mesh wires to crisp dark slate (#1E293B) without
        // snapping strain/heat colormap zero-endpoints (luma <= 0.908).
        if chroma < 0.22 && luma > 0.925 {
            return vec4<f32>(0.118, 0.161, 0.231, in.color.a);
        }
        // Darken pale warm yellow quotient rim wires (>= 5.0:1 on #F4F6FA) without altering
        // sequential_heat_rgba peaks (whose blue channel stays <= 0.22).
        if c.r > 0.95 && c.g > 0.80 && c.b > 0.36 && c.b < 0.55 {
            return vec4<f32>(0.58, 0.34, 0.02, in.color.a);
        }
    } else {
        // Dark Studio backdrop: lift dark neutral atlas wires to cool light slate (#B8C7D9).
        if chroma < 0.20 && luma < 0.38 {
            return vec4<f32>(0.722, 0.780, 0.851, in.color.a);
        }
    }
    return in.color;
}
