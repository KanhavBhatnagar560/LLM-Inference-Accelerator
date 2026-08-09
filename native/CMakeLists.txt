add_library(specdecode_native SHARED src/native.cpp)
add_library(specdecode::native ALIAS specdecode_native)

target_compile_features(specdecode_native PUBLIC cxx_std_17)
set_target_properties(
    specdecode_native
    PROPERTIES
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        VERSION 1.1.0
        SOVERSION 1
)

target_compile_definitions(specdecode_native PRIVATE SD_NATIVE_BUILD)
target_include_directories(
    specdecode_native
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(specdecode_native PRIVATE /W4 /permissive-)
else()
    target_compile_options(specdecode_native PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(
    TARGETS specdecode_native
    EXPORT specdecodeNativeTargets
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
    add_executable(specdecode_native_tests tests/native_tests.cpp)
    target_compile_features(specdecode_native_tests PRIVATE cxx_std_17)
    target_link_libraries(specdecode_native_tests PRIVATE specdecode::native)
    add_test(NAME specdecode_native_tests COMMAND specdecode_native_tests)

    add_executable(specdecode_native_c_header_test tests/header_c_test.c)
    target_link_libraries(specdecode_native_c_header_test PRIVATE specdecode::native)
    add_test(NAME specdecode_native_c_header_test COMMAND specdecode_native_c_header_test)
endif()
