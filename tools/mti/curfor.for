      PROGRAM test_getcwd
      CHARACTER(len=255) :: cwd
      CALL getcwd(cwd)
      WRITE(*,*) TRIM(cwd)
      END PROGRAM