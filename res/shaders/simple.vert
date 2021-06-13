#version 430 core

#define PI 3.1415926538

layout(location = 0) in vec3 position;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 colour;

layout(location = 1) out vec3 out_position;
layout(location = 2) out vec3 out_normal;
layout(location = 3) out vec3 out_colour;

layout(location=12) uniform float scale;

layout(location=16) uniform mat4x4 MVP;
layout(location=20) uniform mat4x4 normalMatrix;
layout(location=24) uniform mat4x4 MV;

layout(location=5) uniform float angleLimit;
layout(location=6) uniform vec3 rotationOrigin;
layout(location=7) uniform vec3 rotationDirection;

void main()
{
    vec3 transformedPosition = position;

    out_position = vec3(MV * vec4(transformedPosition, 1.0));
    out_normal = vec3(normalMatrix * vec4(normal, 1.0));
    out_colour = colour;

    gl_Position = MVP * vec4(transformedPosition, 1.0f);
}
