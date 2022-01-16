enable_testing()

add_executable(Test "headers/framework.h" "framework.cpp" "cases/handler.cpp" "cases/tests.cpp" "cases/utils.cpp")
target_link_libraries(Test Atlus)

add_test(NAME test COMMAND Test)
