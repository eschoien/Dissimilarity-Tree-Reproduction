#version 430 core

layout(location = 1) in vec2 texCoords;

out vec4 color;

layout(binding = 3) uniform sampler2D texture0;

void main()
{
	float value = texture(texture0, texCoords).r;;
	color = vec4(value, value, value, 1.0);
}
