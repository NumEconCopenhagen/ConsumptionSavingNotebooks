namespace index {

int d2(int i_x1, int i_x2, int Nx1, int Nx2){

    int i = i_x1*Nx2
            + i_x2;

    return i;

} // d2

int d3(int i_x1, int i_x2, int i_x3, int Nx1, int Nx2, int Nx3){

    int i = i_x1*Nx2*Nx3
            + i_x2*Nx3
            + i_x3;

    return i;

} // d3

int d4(int i_x1, int i_x2, int i_x3, int i_x4, int Nx1, int Nx2, int Nx3, int Nx4){

    int i = i_x1*Nx2*Nx3*Nx4
            + i_x2*Nx3*Nx4
            + i_x3*Nx4
            + i_x4;

    return i;

} // d4

int d5(int i_x1, int i_x2, int i_x3, int i_x4, int i_x5, int Nx1, int Nx2, int Nx3, int Nx4, int Nx5){

    int i = i_x1*Nx2*Nx3*Nx4*Nx5
            + i_x2*Nx3*Nx4*Nx5
            + i_x3*Nx4*Nx5
            + i_x4*Nx5
            + i_x5;

    return i;

} // d5

} // namespace