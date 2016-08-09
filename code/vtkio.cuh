#ifndef VTK_IO
#define VTKIO

//Functions for IO to disk. Read vtk files, write some other format

float orderSwap( float f );
int vtkDataRead (float4* data, const char* filename, float4 &origin);
void float4write (const char* location, int steps, float4* h_lines);
void floatwrite (const char* location, int steps, float* h_lines);
#endif
