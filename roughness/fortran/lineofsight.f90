SUBROUTINE inputvector(elev,azim,inputV)
  !** This subroutine converts elev and azim to a 3D vector.
  IMPLICIT NONE
  REAL(8), intent(in) :: elev, azim
  REAL(8), intent(inout), dimension(0:2) :: inputV
  REAL(8) :: elevr, azimr
  REAL(8),  parameter :: PI  = 4 * atan (1.0_8)

  elevr = elev * PI/180
  azimr = azim * PI/180

  inputV(0) = sin(azimr)*sin(elevr)
  inputV(1) = -cos(azimr)*sin(elevr)
  inputV(2) = cos(elevr)

END SUBROUTINE inputvector

SUBROUTINE inv_norm_vector(inputV,inv_inputV,norm_inputV)
  !** This subroutine computes the normalized inverse of the input vector
  IMPLICIT NONE
  REAL(8), intent(in), dimension(0:2) :: inputV
  REAL(8), intent(inout), dimension(0:2) :: inv_inputV, norm_inputV

  inv_inputV = -inputV/Maxval(ABS(inputV(0:1)))
  norm_inputV(2) = sqrt(inputV(0)**2+inputV(1)**2)
  norm_inputV(0) = -inputV(0)*inputV(2)/norm_inputV(2)
  norm_inputV(1) = -inputV(1)*inputV(2)/norm_inputV(2)

END SUBROUTINE inv_norm_vector

SUBROUTINE lineofsight(dem, elev, azim, los, cols, rows)
  !**This routine is adapted from the R/arcgis insol module and is modified
  !**to run in python through the NumPy f2py tool.

  IMPLICIT NONE
  INTEGER, intent(in) :: cols, rows
  REAL(8), intent(inout) :: elev, azim
  REAL(8), intent(in), dimension(rows*cols) :: dem
  REAL(8), intent(inout), dimension(0:cols-1, 0:rows-1) :: los
!f2py intent(in,out) :: los
  REAL(8), dimension(0:cols-1, 0:rows-1) :: z
  REAL(8), dimension(0:2) :: inputV,inv_inputV, norm_inputV, vec2origin
  REAL(8) :: dx, dy, zproj, zcomp
  INTEGER :: idx, jdy, n, i, j, f_i, f_j, casx, casy, newshape(2)
  EXTERNAL :: inputvector, inv_norm_vector

  newshape(1)=cols
  newshape(2)=rows
  z=reshape(dem,newshape)

  call inputvector(elev, azim, inputV)
  call inv_norm_vector(inputV,inv_inputV,norm_inputV)

  !** Make the casx integer large enough to compare effectively
  casx=NINT(1e6*inputV(0))
  SELECT CASE (casx)
    !** if case (:0), inputV(x) negative, vector is West: beginning of grid cols
    CASE (:0)
      f_i=0
    !** Otherwise, vector is East: end of grid cols
    CASE default
      f_i=cols-1
  END SELECT

  !** Make the casy integer large enough to compare effectively
  casy=NINT(1e6*inputV(1))
  SELECT CASE (casy)
    !** if case (:0), inputV(x) negative, vector is North: beginning of grid rows
    CASE (:0)
      f_j=0
    !** Otherwise, vector is south: end of grid rows
    CASE default
      f_j=rows-1
  END SELECT

  !******************* Grid scanning *******************************
  !** The array los stores cell binary line of sight value. Input is all 1's.
  !** Here we loop col and row wise to find where lineofsight==0.
  !*****************************************************************
  j=f_j
  DO i=0, cols-1
      n = 0
      zcomp = -HUGE(zcomp) !** initial value lower than any possible zproj
      DO
          dx=inv_inputV(0)*n
          dy=inv_inputV(1)*n
          idx = NINT(i+dx)
          jdy = NINT(j+dy)
          IF ((idx < 0) .OR. (idx >= cols) .OR. (jdy < 0) .OR. (jdy >= rows)) exit
          vec2origin(0) = dx
          vec2origin(1) = dy
          vec2origin(2) = z(idx,jdy)
          zproj = Dot_PRODUCT(vec2origin,norm_inputV)
          IF (zproj < zcomp) THEN
              los(idx,jdy) = 0
              ELSE
              zcomp = zproj
          END IF
          n=n+1
      END DO
  END DO

  i=f_i
  DO j=0,rows-1
      n = 0
      zcomp = -HUGE(zcomp)  !** initial value lower than any possible zproj
      DO
          dx=inv_inputV(0)*n
          dy=inv_inputV(1)*n
          idx = NINT(i+dx)
          jdy = NINT(j+dy)
          IF ((idx < 0) .OR. (idx >= cols) .OR. (jdy < 0) .OR. (jdy >= rows)) exit
          vec2origin(0) = dx
          vec2origin(1) = dy
          vec2origin(2) = z(idx,jdy)
          zproj = Dot_PRODUCT(vec2origin,norm_inputV)
          IF (zproj < zcomp) THEN
              los(idx,jdy) = 0
              ELSE
              zcomp = zproj
          END IF
          n=n+1
      END DO
  END DO
END SUBROUTINE lineofsight
