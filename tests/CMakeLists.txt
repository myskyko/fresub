cmake_minimum_required(VERSION 3.18)

# Tests - flattened structure
set(TEST_SOURCES
    test_window_extraction.cpp
    test_simulation.cpp
    test_feasibility.cpp
    test_synthesis.cpp
    test_insertion.cpp
    test_blif_aig_loader.cpp
)

# Add test executables
foreach(test_source IN LISTS TEST_SOURCES)
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_include_directories(${test_name} PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(${test_name} 
        fresub_cpu 
        Threads::Threads
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
