include(FetchContent)
set(FETCH_PACKAGES "")

FetchContent_Declare(
  mqt-core
  GIT_REPOSITORY https://github.com/munich-quantum-toolkit/core.git
  GIT_TAG 13bf566d69ed8923458a085e854b24df976a4103)
list(APPEND FETCH_PACKAGES mqt-core)

FetchContent_MakeAvailable(${FETCH_PACKAGES})
