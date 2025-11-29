include(FetchContent)

# Silence or keep verbose:
# set(FETCHCONTENT_QUIET OFF)

# CMake 3.24+ policy to fix timestamp warning (see #2 below)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE   # also fixes the warning
)
FetchContent_MakeAvailable(googletest)

# Then link it to your tests target:
# target_link_libraries(unit_tests PRIVATE gtest_main gmock)
