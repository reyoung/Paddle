INCLUDE(ExternalProject)

SET(CONCURRENT_QUEUE_SOURCE_DIR ${THIRD_PARTY_PATH}/concurrentqueue)
SET(CONCURRENT_INCLUDE_DIR ${CONCURRENT_QUEUE_SOURCE_DIR}/src/extern_concurrentqueue)
message(status ${CONCURRENT_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CONCURRENT_INCLUDE_DIR})

ExternalProject_Add(
        extern_concurrentqueue
        ${EXTERNAL_PROJECT_LOG_ARGS}
        GIT_REPOSITORY  "https://github.com/cameron314/concurrentqueue.git"
        GIT_TAG         v1.0.0-beta
        PREFIX          ${CONCURRENT_QUEUE_SOURCE_DIR}
        UPDATE_COMMAND  ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
)
if (${CMAKE_VERSION} VERSION_LESS "3.3.0")
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/eigen3_dummy.c)
    file(WRITE ${dummyfile} "const char *dummy_eigen3 = \"${dummyfile}\";")
    add_library(concurrentqueue STATIC ${dummyfile})
else()
    add_library(concurrentqueue INTERFACE)
endif()

add_dependencies(concurrentqueue extern_concurrentqueue)

LIST(APPEND external_project_dependencies concurrentqueue)
