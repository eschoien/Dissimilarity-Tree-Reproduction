#define GLM_ENABLE_EXPERIMENTAL
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <chrono>
#include <vector>
#include <iostream>
#include <shapeDescriptor/cpu/types/float4.h>
#include <shapeDescriptor/libraryBuildSettings.h>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include "SymmetryVisualiser.h"
#include "shader.hpp"
#include "VAOGenerator.h"
#include "GLUtils.h"

struct PositionInfo {
    glm::vec3 position = {0, 0, 0};
    glm::vec3 rotation = {0, 0, 0};
    float scale = 1;

    PositionInfo() {}
    PositionInfo(glm::vec3 position, glm::vec3 rotation, float scale) : position(position), rotation(rotation), scale(scale) {}
};

void handleKeyboardInput(GLFWwindow* window)
{
    // Use escape key for terminating the GLFW window
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

static std::chrono::steady_clock::time_point _previousTimePoint = std::chrono::steady_clock::now();

// Calculates the elapsed time since the previous time this function was called.
double getTimeDeltaSeconds() {
    // Determine the current time
    std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();

    // Calculate the number of nanoseconds that elapsed since the previous call to this function
    long long timeDelta = std::chrono::duration_cast<std::chrono::nanoseconds>(currentTime - _previousTimePoint).count();
    // Convert the time delta in nanoseconds to seconds
    double timeDeltaSeconds = (double)timeDelta / 1000000000.0;

    // Store the previously measured current time
    _previousTimePoint = currentTime;

    // Return the calculated time delta in seconds
    return timeDeltaSeconds;
}

void visualise(ShapeDescriptor::cpu::Mesh mesh,
               ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::SearchResults<unsigned int>> searchResults,
               float scale, ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptors) {
    GLFWwindow* window = GLinitialise();

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glm::vec3 selectedVertex = {0, 0, 0};
    unsigned int selectedVertexIndex = 0;

    ShapeDescriptor::cpu::float3 indicatorVertices[] = {
            {-1, 0, 0},
            {1, 0, 0},
            {0, -1, 0},
            {0, 1, 0},
            {0, 0, -1},
            {0, 0, 1}};
    ShapeDescriptor::cpu::float3 indicatorNormals[] = {
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0}};
    ShapeDescriptor::cpu::float3 indicatorColours[] = {
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0}};


    BufferObject indicatorVAO = generateVertexArray(indicatorVertices, indicatorNormals, indicatorColours, 6);

    ShapeDescriptor::cpu::float3* meshColours = new ShapeDescriptor::cpu::float3[mesh.vertexCount];
    const ShapeDescriptor::cpu::float3 defaultColour = {0.5, 0.5, 0.5};
    std::fill(meshColours, meshColours + mesh.vertexCount, defaultColour);
    BufferObject meshVAO = generateVertexArray(mesh.vertices, mesh.normals, meshColours, mesh.vertexCount);

    Shader shader;
    shader.attach("../res/shaders/simple.vert");
    shader.attach("../res/shaders/simple.frag");
    shader.link();

    Shader singleChannelShader;
    singleChannelShader.attach("../res/shaders/singlechannel.vert");
    singleChannelShader.attach("../res/shaders/singlechannel.frag");
    singleChannelShader.link();

    // Set default colour after clearing the colour buffer
    glClearColor(0.3f, 0.5f, 0.8f, 1.0f);

    PositionInfo cameraPosition;

    const glm::vec3 zAxis(0, 0, 1);
    const glm::vec3 yAxis(0, 1, 0);
    const glm::vec3 xAxis(1, 0, 0);

    const float rotationSpeed = 0.08f;
    const float cameraSpeed = 0.03f;

    std::vector<glm::vec3> stageLights = {
            //{0,    1, 0},
            //{-1, 1, -1},
            //{1,  1, -1},
            //{-1, 1, 1},
            //{1,  1, 1},
            {0, 0, 0}
    };
    std::vector<glm::vec3> transformedStagelights(32);

    bool leftMouseWasDown = false;

    // Rendering Loop
    while (!glfwWindowShouldClose(window)) {
        int axisCount;
        const float *axes = glfwGetJoystickAxes(GLFW_JOYSTICK_1, &axisCount);
        int buttonCount;
        const unsigned char *buttons = glfwGetJoystickButtons(GLFW_JOYSTICK_1, &buttonCount);

        float frameTime = getTimeDeltaSeconds();

        int windowWidth, windowHeight;
        glfwGetWindowSize(window, &windowWidth, &windowHeight);

        glViewport(0, 0, windowWidth, windowHeight);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // --- Handle Input --

        float deltaX = axes[0];
        float deltaY = -1 * axes[1];
        float deltaRotationX = std::abs(axes[3]) > 0.3 ? axes[3] : 0;
        float deltaRotationY = std::abs(axes[4]) > 0.3 ? -1.0f * axes[4] : 0;

        if (std::abs(deltaRotationX) < 0.15) { deltaRotationX = 0; }
        if (std::abs(deltaRotationY) < 0.15) { deltaRotationY = 0; }
        if (std::abs(deltaX) < 0.15) { deltaX = 0; }
        if (std::abs(deltaY) < 0.15) { deltaY = 0; }

        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            cameraPosition.rotation.x -= rotationSpeed;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            cameraPosition.rotation.x += rotationSpeed;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            cameraPosition.rotation.y -= rotationSpeed;
        }
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            cameraPosition.rotation.y += rotationSpeed;
        }
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
            cameraPosition.rotation.z -= rotationSpeed;
        }
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
            cameraPosition.rotation.z += rotationSpeed;
        }

        if (std::abs(deltaX) < 0.1) {
            deltaX = 0;
        }

        if (std::abs(deltaY) < 0.1) {
            deltaY = 0;
        }

        cameraPosition.rotation.x += deltaRotationX * rotationSpeed;
        cameraPosition.rotation.y -= deltaRotationY * rotationSpeed;

        float angleYRadiansForward = cameraPosition.rotation.x;
        float angleYRadiansSideways = (cameraPosition.rotation.x + float(M_PI / 2.0));

        cameraPosition.position.x -= deltaY * std::sin(angleYRadiansForward) * cameraSpeed;
        cameraPosition.position.z += deltaY * std::cos(angleYRadiansForward) * cameraSpeed;

        cameraPosition.position.x -= deltaX * std::sin(angleYRadiansSideways) * cameraSpeed;;
        cameraPosition.position.z += deltaX * std::cos(angleYRadiansSideways) * cameraSpeed;

        cameraPosition.position.y += (((axes[2] + 1.0f) / 2.0f) - ((axes[5] + 1.0f) / 2.0f)) * cameraSpeed;

        shader.activate();

        glm::mat4x4 view(1.0);

        view *= glm::rotate(cameraPosition.rotation.z, zAxis);
        view *= glm::rotate(cameraPosition.rotation.y, xAxis);
        view *= glm::rotate(cameraPosition.rotation.x, yAxis);
        view *= glm::translate(cameraPosition.position);

        glm::mat4x4 projection = glm::perspective(1.57f, (float) windowWidth / (float) windowHeight, 0.001f, 100.0f);
        glm::mat4x4 viewProjection = projection * view;

        stageLights.at(stageLights.size() - 1) = -cameraPosition.position;

        for (int i = 0; i < stageLights.size(); i++) {
            transformedStagelights.at(i) = view * glm::vec4(stageLights.at(i), 1.0);
        }

        glProgramUniform3fv(shader.get(), 32, transformedStagelights.size(),
                            reinterpret_cast<const GLfloat *>(transformedStagelights.data()));
        glUniform1i(31, stageLights.size());

        glm::vec3 meshRotation = {0, 0, 0};
        glm::vec3 meshPosition = {0, 0, 0};

        glm::mat4 sceneObjectRotations =
                glm::rotate(glm::radians(meshRotation.x), glm::vec3(1, 0, 0)) *
                glm::rotate(glm::radians(meshRotation.y), glm::vec3(0, 1, 0)) *
                glm::rotate(glm::radians(meshRotation.z), glm::vec3(0, 0, 1));

        glm::mat4 sceneObjectTransform =
                glm::translate(meshPosition) *
                sceneObjectRotations *
                glm::scale(glm::vec3(scale, scale, scale));


        glUniformMatrix4fv(16, 1, GL_FALSE, glm::value_ptr(viewProjection * sceneObjectTransform));
        glUniformMatrix4fv(20, 1, GL_FALSE,
                           glm::value_ptr(glm::transpose(glm::inverse(view * sceneObjectRotations))));
        glUniformMatrix4fv(24, 1, GL_FALSE, glm::value_ptr(view * sceneObjectTransform));

        glBindVertexArray(meshVAO.VAOID);
        glDrawElements(GL_TRIANGLES, mesh.vertexCount, GL_UNSIGNED_INT, 0);


        // Indicator

        float depth = 0;
        double mouseX, mouseY;
        glfwGetCursorPos(window, &mouseX, &mouseY);
        glReadPixels(int(mouseX), int(windowHeight - mouseY), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
        depth = depth * 2.0 - 1.0;
        glm::vec4 clipSpaceCoordinates = {2.0 * (mouseX / float(windowWidth)) - 1.0,
                                          2.0 * (1.0 - (mouseY / float(windowHeight))) - 1.0,
                                          depth, 1.0};
        glm::vec4 unprojectedCoordinates = glm::inverse(viewProjection * sceneObjectTransform) * clipSpaceCoordinates;
        unprojectedCoordinates /= unprojectedCoordinates.w;
        glm::vec3 indicatorPosition = glm::vec3(unprojectedCoordinates);
        glm::mat4 indicatorPositioningMatrix = glm::translate(glm::mat4(1.0), selectedVertex);

        glUniformMatrix4fv(16, 1, GL_FALSE, glm::value_ptr(viewProjection * sceneObjectTransform * indicatorPositioningMatrix));
        glUniformMatrix4fv(24, 1, GL_FALSE, glm::value_ptr(view * sceneObjectTransform * indicatorPositioningMatrix));

        glBindVertexArray(indicatorVAO.VAOID);
        glDrawElements(GL_LINES, 6, GL_UNSIGNED_INT, 0);

        bool leftMouseIsDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
        if(leftMouseIsDown/* && !leftMouseWasDown*/) {
            size_t closestVertexIndex = 0;
            float closestDistance = FLT_MAX;

            for(size_t i = 0; i < mesh.vertexCount; i++) {
                ShapeDescriptor::cpu::float3 vertex = mesh.vertices[i];
                glm::vec3 delta = glm::vec3(vertex.x, vertex.y, vertex.z) - indicatorPosition;
                float distance = glm::length(delta);

                if(distance < closestDistance) {
                    closestDistance = distance;
                    closestVertexIndex = i;
                }
            }

            selectedVertexIndex = closestVertexIndex;
            ShapeDescriptor::cpu::float3 vertex = mesh.vertices[selectedVertexIndex];
            selectedVertex = glm::vec3(vertex.x, vertex.y, vertex.z);

            std::fill(meshColours, meshColours + mesh.vertexCount, defaultColour);

            /*std::cout << "+";
            for(int i = 0; i < spinImageWidthPixels; i++) {
                std::cout << "-";
            }
            std::cout << "+" << std::endl;
            for(int row = spinImageWidthPixels - 1; row >= 0; row--) {
                std::cout << "|";
                for(int col = 0; col < spinImageWidthPixels; col++) {
                    unsigned int chunkIndex = 2 * row + col / 32;
                    unsigned int chunk = descriptors.content[selectedVertexIndex].contents[chunkIndex];
                    bool pixelValue = ((chunk >> (31U - (col % 32U))) & 0x1U) == 0x1U;
                    std::cout << (pixelValue ? "X" : " ");
                }
                std::cout << "|" << std::endl;
            }
            std::cout << "+";
            for(int i = 0; i < spinImageWidthPixels; i++) {
                std::cout << "-";
            }
            std::cout << "+" << std::endl;
*/
            for(unsigned int i = 0; i < mesh.vertexCount; i++) {
                float distance = length(ShapeDescriptor::cpu::float3(selectedVertex.x, selectedVertex.y, selectedVertex.z) - mesh.vertices[i]);
                if(distance < 0.12) {
                    meshColours[i].y = 0.8;
                } else {
                    meshColours[i].y = 0.5;
                }
            }

            ShapeDescriptor::gpu::SearchResults<unsigned int> results = searchResults.content[selectedVertexIndex - 74958];

            for(unsigned int searchResultIndex = 0; searchResultIndex < 1; searchResultIndex++) {
                float redLevel = std::max<float>(0, float(255 - (3 * int(results.scores[searchResultIndex]))) / 255.0);
                meshColours[results.indices[searchResultIndex]] = {redLevel, 0, 0};
            }

            glDeleteVertexArrays(1, &meshVAO.VAOID);
            glDeleteBuffers(1, &meshVAO.indexBufferID);
            glDeleteBuffers(1, &meshVAO.vertexBufferID);
            glDeleteBuffers(1, &meshVAO.normalBufferID);
            glDeleteBuffers(1, &meshVAO.colourBufferID);
            meshVAO = generateVertexArray(mesh.vertices, mesh.normals, meshColours, mesh.vertexCount);
        }



        leftMouseWasDown = leftMouseIsDown;

        // Handle other events
        glfwPollEvents();
        handleKeyboardInput(window);

        // Flip buffers
        glfwSwapBuffers(window);
    }
}
