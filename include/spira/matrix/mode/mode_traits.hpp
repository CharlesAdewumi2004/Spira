#include <cstddef>

#include <spira/matrix/mode/matrix_mode.hpp>
#include <spira/config.hpp>
namespace spira::mode
{

    static constexpr config::mode_policy policy_for(matrix_mode m)
    {
        switch (m)
        {
        case matrix_mode::spmv:
            return config::spmv;
        case matrix_mode::balanced:
            return config::balanced;
        case matrix_mode::insert_heavy:
            return config::insert_heavy;
        default:
            throw std::invalid_argument("Invalid mode passed");
        }
    }

}