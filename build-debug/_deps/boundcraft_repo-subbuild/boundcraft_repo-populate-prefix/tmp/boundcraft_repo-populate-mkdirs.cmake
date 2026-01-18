# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/charlie/dev/spira/build-debug/_deps/boundcraft_repo-src"
  "/home/charlie/dev/spira/build-debug/_deps/boundcraft_repo-build"
  "/home/charlie/dev/spira/build-debug/_deps/boundcraft_repo-subbuild/boundcraft_repo-populate-prefix"
  "/home/charlie/dev/spira/build-debug/_deps/boundcraft_repo-subbuild/boundcraft_repo-populate-prefix/tmp"
  "/home/charlie/dev/spira/build-debug/_deps/boundcraft_repo-subbuild/boundcraft_repo-populate-prefix/src/boundcraft_repo-populate-stamp"
  "/home/charlie/dev/spira/build-debug/_deps/boundcraft_repo-subbuild/boundcraft_repo-populate-prefix/src"
  "/home/charlie/dev/spira/build-debug/_deps/boundcraft_repo-subbuild/boundcraft_repo-populate-prefix/src/boundcraft_repo-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/charlie/dev/spira/build-debug/_deps/boundcraft_repo-subbuild/boundcraft_repo-populate-prefix/src/boundcraft_repo-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/charlie/dev/spira/build-debug/_deps/boundcraft_repo-subbuild/boundcraft_repo-populate-prefix/src/boundcraft_repo-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
