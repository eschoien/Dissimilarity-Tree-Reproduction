#version 430 core

layout(location = 1) in vec3 vertexCoord;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 colour;

layout(location = 31) uniform int activeLightCount;
layout(location = 32) uniform vec3 lightPositions[16];


const vec3 eyePosition = vec3(0, 0, 0);
const float specularStrength = 0.5;

out vec4 color;

vec3 computeColourForLightSource(vec3 surfaceColour, vec3 lightSourcePosition, vec3 normalisedNormal, vec3 surfaceToEyeVector) {
	float distanceToLightSource = length(lightSourcePosition - vertexCoord);
	vec3 surfaceToLightVector = normalize(lightSourcePosition - vertexCoord);

	float attenuation = clamp(1.0 - distanceToLightSource*distanceToLightSource/ (9 * 9), 0.0, 1.0);
	attenuation *= attenuation;

	float diffuse = max(dot(normalisedNormal, surfaceToLightVector) * attenuation, 0);
	vec3 diffuseColour = diffuse * surfaceColour;

	float specular = pow(max(dot(surfaceToEyeVector, reflect(-surfaceToLightVector, normalisedNormal)), 0), 4) * attenuation;
	vec3 specularColour = specular * vec3(0.3, 0.3, 0.3);

	return diffuseColour + specularColour;
}

void main()
{
	vec3 normalisedNormal = normalize(normal);

	vec4 diffuseColour = vec4(colour, 1);

	vec3 surfaceToEyeVector = normalize(eyePosition - vertexCoord);

	vec3 finalColour = vec3(0, 0, 0);

	for(int i = 0; i < activeLightCount; i++) {
		finalColour += computeColourForLightSource(diffuseColour.xyz, lightPositions[i], normalisedNormal, surfaceToEyeVector);
	}

	color = vec4(finalColour, diffuseColour.a);
}
