#include "vtkio.h"
#include <stdio.h>
#include "constants.h"
#include <iostream>
#include <cstring>

//Swap order of bytes to convert from little-endian to big-endian
float orderSwap( float f )
{
	union
	{
		float f;
	    unsigned char b[4];
	} dat1, dat2;

    dat1.f = f;
	dat2.b[0] = dat1.b[3];
	dat2.b[1] = dat1.b[2];
	dat2.b[2] = dat1.b[1];
	dat2.b[3] = dat1.b[0];
	return dat2.f;
}

/*	Reads the data from filename (in vtk format) and stores the origin in the file at origin
	and the vector field in the file in data
*/
int vtkDataRead (float4* data, const char* filename, float4 &origin) {
	unsigned int datasize;
	unsigned int n_x =0, n_y=0, n_z=0;
	char kind[20];
	char name[20];
	char type[20];
	char rstr[80];
	float4 rspacing;
	FILE *dfp;

	dfp= fopen(filename, "r");
	for(unsigned int i=0; i<4; ++i) {
		fgets(rstr, 80, dfp);
		std::cout << rstr;
	}
	fscanf(dfp, "%s %u %u %u", rstr, &n_x, &n_y, &n_z);
	if(!( n_x == N && n_y == N && n_z == N)) {
		std::cout<<"Warning: incorrect " << rstr << " read: expected 256, got: " << n_z << std::endl;
		fclose(dfp);
		return 1;
	}
	fscanf(dfp, "%s %f %f %f", rstr, &origin.x, &origin.y, &origin.z);
	fscanf(dfp, "%s %f %f %f", rstr, &rspacing.x, &rspacing.y, &rspacing.z);
	if(! (rspacing.x == rspacing.y && rspacing.y == rspacing.z)) {
		std::cout << "Warning: (unsupported) anisotrope spacing read!" << std::endl;
		fclose(dfp);
	}
	fscanf(dfp, "%s %u", rstr, &datasize);
	if(datasize != n_x*n_y*n_z) {
		std::cout<<"Error: " << rstr << "is not equal to n_x*n_y*n_z" << std::endl;
		fclose(dfp);
		return 1;
	}
	fscanf(dfp, "%s %s %s\n", kind, name, type);
	while(strcmp(name,"bfield")!=0) {
		if(strcmp(kind,"SCALARS")==0) {
			printf("Found a SCALAR field %s, discarding it", name);
		    fgets(rstr, 80, dfp);
			float todiscard = 0;
			for(unsigned int i=0; i<datasize; ++i) {
				fread(&todiscard, sizeof(float), 1, dfp);
			}
		} else if (strcmp(kind,"VECTORS")==0) {
			printf("Found a VECTOR field %s, discarding it", name);
			float todiscard[3] = {0,0,0};
			for(unsigned int i=0; i<datasize; ++i) {
				fread(&todiscard, sizeof(float), 3, dfp);
			}
		} else {
			printf("Error: unknown datatype %s", name);
			return 1;
		}
		fscanf(dfp, "%s %s %s\n", kind, name, type);
		printf("Je moeder is een correct: %s, %s, %s\n",kind, name, type);
	}
	if(strcmp(type,"float")!=0) {
		printf("Error: Incorrect type, found: %s, %s, %s\n",kind, name, type);
		fclose(dfp);
		return 1;
	}
	for(unsigned int i=0; i<datasize; ++i) {
		float datapoint[3] = {0,0,0};
		fread(datapoint, sizeof(float), 3, dfp);
		data[i] = make_float4(orderSwap(datapoint[0]),orderSwap(datapoint[1]),orderSwap(datapoint[2]),0);
	}
	std::cout << "Data read in was successful!" << std::endl;
	fclose(dfp);
	return 0;
}


void datawrite (const char* location, int steps, float4* h_lines){ 
    //write the first streamline to a file. Remember this is 32 bits when reading!
    FILE *fp;
    fp = fopen(location, "w");
    for (unsigned int i = 0; i<steps; i++){   //write only the first streamline
        fwrite(&h_lines[i], sizeof(float4), 1, fp);
    }
    fclose(fp);
    std::cout<<"streamline written!"<<std::endl;
}

