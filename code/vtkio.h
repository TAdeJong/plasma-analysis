#ifndef VTK_IO
#define VTKIO

//Functions for IO to disk. Read vtk files, write some other format

float orderSwap( float f );
int vtkDataRead (float4* data, const char* filename, float4 &origin);
void datawrite (const char* location, int steps, float4* h_lines);
#endif
