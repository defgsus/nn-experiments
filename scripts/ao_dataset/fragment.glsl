#version 330

in vec2 v_pos;
in vec2 v_uv;

out vec4 fragColor;

uniform ivec2 u_resolution;
uniform sampler2D u_texture;


float DE(in vec3 pos) {
    vec2 uv = pos.xy * .5 + .5;

    // distance from input texture (distance, blurred distance, mask, blurred mask)
    vec4 t = texture(u_texture, uv).xyzw;
    // cut at z
    float f = max(t.x * 3., dot(pos - vec3(0, 0, .2 * (.2+.8*t.w)), vec3(0, 0, 1)));

    float d = dot(pos, vec3(0, 0, 1));
    d = min(d, f);
    d += min(0., t.y + t.x) / 4.;

    //d = min(d, length(pos) - 1.);

    return d;
}

vec3 NORM(in vec3 pos) {
    vec2 e = vec2(0.001, 0);
    return normalize(vec3(
        DE(pos + e.xyy) - DE(pos - e.xyy),
        DE(pos + e.yxy) - DE(pos - e.yxy),
        DE(pos + e.yyx) - DE(pos - e.yyx)
    ));
}

const vec3 light_norm = normalize(vec3(1, 1, 3));

float light_at(in vec3 pos) {
    const float k = 2.;

    float t = 0.001;
    float light = 1.;
    for (int i=0; i < 250; ++i) {
        vec3 p = pos + light_norm * t;
        float d = DE(p);
        if (d <= 0.0001) {
            return 0.;
        }
        light = min(light, k * d / t);
        t += d * .1;
    }
    return light;
}

float ambient_at(in vec3 pos, in vec3 dir) {
    const float k = 1.5;

    float t = 0.01;
    float light = 1.;
    for (int i=0; i < 250; ++i) {
        vec3 p = pos + dir * t;
        if (p.x<=-1 || p.x>=1 || p.y<=-1 || p.y>=1) break;
        float d = DE(p);
        if (d <= 0.001) {
            return 0.;
        }
        light = min(light, k * d / t);
        t += d * .4;
    }
    float soft = pow(light, 2.);
    float hard = pow(light, 1/2.);
    return hard * .7 + .3 * soft;
}

vec3 rotate_z(in vec3 p, in float r) {
    return vec3(
        p.x * cos(r) - p.y * sin(r),
        p.x * sin(r) + p.y * cos(r),
        p.z
    );
}


vec3 render(in vec3 pos, in vec3 dir) {
    float t = 0;
    for (int i=0; i < 50; ++i) {
        vec3 p = pos + dir * t;
        float d = DE(p);
        if (d <= 0.001) {
            break;
        }
        t += d;
    }

    vec3 p = pos + dir * t;

    vec3 norm = NORM(p);
    float light = light_at(p);

    vec3 adir = reflect(dir, norm);
    float f = .7;
    float amb = (
          ambient_at(p, normalize(adir + vec3(-f, 0, 0)))
        + ambient_at(p, normalize(adir + vec3( f, 0, 0)))
        + ambient_at(p, normalize(adir + vec3(0, f, 0)))
        + ambient_at(p, normalize(adir + vec3(0, -f, 0)))
    )/4.;

    vec4 tex = texture(u_texture, p.xy * .5 + .5);
    //vec3 col = norm * .5 + .5;
    vec3 col = vec3(.2-.2*tex.x, .4-.2*tex.y, .3-.2*tex.w);
    col = rotate_z(col, tex.w+amb);
    col = rotate_z(col.yzx, -tex.y*2.).zxy;
    col = rotate_z(rotate_z(col, -1.2).yzx, -1.3).zxy;
    col *= .6*vec3(3, 2, 1);  //mix(col.xy, col.yx, .8);

    col = mix(col, vec3(.5), tex.z);

    float phong = max(0., dot(norm, normalize(light_norm * vec3(1, 1, .4))));
    col += pow(phong, 7.) * (.2 + .8 * light);

    col *= .5 + .5 * light;
    col *= .5 + .5 * amb;


    col = mix(col, vec3(col.r*.3 + col.g*.6 + col.b*.1), .5);
    return clamp(col, 0, 1);
}

void main() {

    vec3 pos = vec3(v_pos, 1);
    vec3 ray = vec3(0, 0, -1);

    vec3 col = render(pos, ray);

    fragColor = vec4(col, 1);
}
