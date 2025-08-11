texture<float, 3, cudaReadModeElementType> im_PSF; 

#define _X ( threadIdx.x + blockIdx.x * blockDim.x)
#define _Y ( threadIdx.y + blockIdx.y * blockDim.y)
#define _Z ( threadIdx.z + blockIdx.z * blockDim.z)

#define __WIDTH ( blockDim.x * gridDim.x )
#define __HEIGHT ( blockDim.y * gridDim.y )
#define __DEPTH ( blockDim.z * gridDim.z )

#define GID_3D (_X + _Y * __WIDTH + _Z * __WIDTH * __HEIGHT) 

__device__ float3 operator+(const float3 &a, const float3 &b) 
{return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);}

__device__ float3 operator+(const float &a, const float3 &b)
{return make_float3(a+b.x, a+b.y, a+b.z);}

__device__ float3 operator+(const float3 &a, const float &b) 
{return make_float3(a.x+b, a.y+b, a.z+b);}

__device__ float3 operator-(const float3 &a, const float3 &b) 
{return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);}

__device__ float3 operator-(const float &a, const float3 &b) 
{return make_float3(a-b.x, a-b.y, a-b.z);}

__device__ float3 operator-(const float3 &a, const float &b) 
{return make_float3(a.x-b, a.y-b, a.z-b);}

__device__ float3 operator*(const float3 &a, const float3 &b) 
{return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);}

__device__ float3 operator*(const float &a, const float3 &b) 
{return make_float3(a*b.x, a*b.y, a*b.z);}

__device__ float3 operator/(const float3 &a, const float3 &b) 
{return make_float3(a.x/b.x, a.y/b.y, a.z/b.z);}

__device__ float3 operator/(const float &a, const float3 &b) 
{return make_float3(a/b.x, a/b.y, a/b.z);}

__device__ float3 floor3(const float3 &a)
{return make_float3(floor(a.x),floor(a.y),floor(a.z));}



__device__ float get_w0(float frac)
{
float fm = 1.f - frac;
return 1.f/6.f * (fm * fm * fm);//
}
__device__ float get_w1(float frac)
{
return 2.f / 3.f - 0.5f * frac * frac * (2.f - frac);
}__device__ float get_w2(float frac)
{
float fm = 1.f - frac;
return 2.f / 3.f - 0.5f * fm * fm * (1.f + frac);
}__device__ float get_w3(float frac)
{
return 1.f/6.f * (frac * frac * frac);
}


__device__ float get_w0_diff(float frac)
{float fm  = 1.f - frac;
return -0.5f * fm * fm;}

__device__ float get_w1_diff(float frac)
{return 1.5f * frac * frac - 2.f * frac;}

__device__ float get_w2_diff(float frac)
{return - 1.5f * frac * frac + frac + 0.5f;}

__device__ float get_w3_diff(float frac)
{return 0.5f * frac * frac;}

__device__ float interp_tricubic(
cudaTextureObject_t im_PSF,
float3 coords,
int deriv)
{
const float3 index = floor3(coords);
const float3 fraction = coords - index;
float w0x,w0y,w0z; 
float w1x,w1y,w1z;
float w2x,w2y,w2z;
float w3x,w3y,w3z;

if (deriv == 0)
    {
    w0x = get_w0(fraction.x), w0y = get_w0(fraction.y), w0z = get_w0(fraction.z);
    w1x = get_w1(fraction.x), w1y = get_w1(fraction.y), w1z = get_w1(fraction.z);
    w2x = get_w2(fraction.x), w2y = get_w2(fraction.y), w2z = get_w2(fraction.z);
    w3x = get_w3(fraction.x), w3y = get_w3(fraction.y), w3z = get_w3(fraction.z);
    }
else if (deriv == 1)
    {
    w0x = get_w0_diff(fraction.x), w0y = get_w0(fraction.y), w0z = get_w0(fraction.z);
    w1x = get_w1_diff(fraction.x), w1y = get_w1(fraction.y), w1z = get_w1(fraction.z);
    w2x = get_w2_diff(fraction.x), w2y = get_w2(fraction.y), w2z = get_w2(fraction.z);
    w3x = get_w3_diff(fraction.x), w3y = get_w3(fraction.y), w3z = get_w3(fraction.z);
    }
else if (deriv == 2)    
    {
    w0x = get_w0(fraction.x), w0y = get_w0_diff(fraction.y), w0z = get_w0(fraction.z);
    w1x = get_w1(fraction.x), w1y = get_w1_diff(fraction.y), w1z = get_w1(fraction.z);
    w2x = get_w2(fraction.x), w2y = get_w2_diff(fraction.y), w2z = get_w2(fraction.z);
    w3x = get_w3(fraction.x), w3y = get_w3_diff(fraction.y), w3z = get_w3(fraction.z);
    }
else
    {
    w0x = get_w0(fraction.x), w0y = get_w0(fraction.y), w0z = get_w0_diff(fraction.z);
    w1x = get_w1(fraction.x), w1y = get_w1(fraction.y), w1z = get_w1_diff(fraction.z);
    w2x = get_w2(fraction.x), w2y = get_w2(fraction.y), w2z = get_w2_diff(fraction.z);
    w3x = get_w3(fraction.x), w3y = get_w3(fraction.y), w3z = get_w3_diff(fraction.z);
    }

float3 w0 = make_float3(w0x,w0y,w0z);
float3 w1 = make_float3(w1x,w1y,w1z);
float3 w2 = make_float3(w2x,w2y,w2z);
float3 w3 = make_float3(w3x,w3y,w3z);

const float3 g0 = w0 + w1;
const float3 g1 = w2 + w3;
const float3 h0 = (w1 / g0) - 0.5f + index;
const float3 h1 = (w3 / g1) + 1.5f + index;

float tex000 = tex3D<float>(im_PSF, h0.x, h0.y, h0.z);
float tex100 = tex3D<float>(im_PSF, h1.x, h0.y, h0.z);
tex000 = g0.x * tex000 + g1.x * tex100;
float tex010 = tex3D<float>(im_PSF, h0.x, h1.y, h0.z);
float tex110 = tex3D<float>(im_PSF, h1.x, h1.y, h0.z);
tex010 = g0.x * tex010 + g1.x * tex110;
tex000 = g0.y * tex000 + g1.y * tex010;
float tex001 = tex3D<float>(im_PSF, h0.x, h0.y, h1.z);
float tex101 = tex3D<float>(im_PSF, h1.x, h0.y, h1.z);
tex001 = g0.x * tex001 + g1.x * tex101;
float tex011 = tex3D<float>(im_PSF, h0.x, h1.y, h1.z);
float tex111 = tex3D<float>(im_PSF, h1.x, h1.y, h1.z);
tex011 = g0.x * tex011 + g1.x * tex111;
tex001 = g0.y * tex001 + g1.y * tex011;
return (g0.z * tex000 + g1.z * tex001);
}

__device__ float get_pixel_sum(
cudaTextureObject_t im_PSF,
float xc,
float yc,
float zc,
int deriv,
int os)
{float out = 0.f;
float offset = 0.f;// ((float)os - 1.f) / 2.f;
float xi = 0.f, yi = 0.f;
for (int ix = 0; ix < os; ix++)
	{xi = xc * (float)os + (float)ix + offset;
	for (int iy = 0; iy < os; iy++) 
		{yi = yc * (float)os + (float)iy + offset;
		out += interp_tricubic(im_PSF,make_float3(zc,yi,xi),deriv);
		}
	}
return out;
}
 

__device__ float clamp(float x, float xmin, float xmax)
{
  return min(xmax,max(xmin, x));
}

__device__ float get_I(
cudaTextureObject_t im_PSF,
float xc,
float yc,
float zc,
float N,
float M_modif,
int deriv,
int os
)
{
//return M_modif + BG + N * interp_tricubic(make_float3(zc,yc,xc),deriv);
return M_modif+ N * get_pixel_sum(im_PSF,xc,yc,zc,deriv,os);
}

__device__ float get_error_poisson(
float I,
float M)
{
float t1 = I - M ;
float t2 = 0.f;
if ((M > 1e-5) & (I > 1e-5))
	{
	t2 =-M * log(I / M);
	}
return max(0.f,t1 + t2);
}


extern "C" __global__ void comp_J(
cudaTextureObject_t im_PSF,
float * J,
float * residual,
float * err_poi,
float * M,
float * x0,
float * y0,
float * z0,
float * BG,
float * NP,
float * M_modif,
float * fac_jac,
int os,
float nz_psf,
int npar)
{
float x0_loc = x0[_Z], y0_loc = y0[_Z], z0_loc = z0[_Z];
float w = __WIDTH, h = __HEIGHT;
float x = (_X - x0_loc), y = (_Y - y0_loc), z = nz_psf/2.f - z0_loc;
float BG_loc = BG[_Z],NP_loc = NP[_Z], M_modif_loc = M_modif[_Z];
float M_loc = M[GID_3D];
float I0 = BG_loc + get_I(im_PSF,x,y,z,NP_loc,M_modif_loc,0,os);

int _XY = _X + __WIDTH * _Y;
int GID_Jx = 0 + _XY * npar + _Z * __WIDTH * __HEIGHT * npar;
int GID_Jy = 1 + _XY * npar + _Z * __WIDTH * __HEIGHT * npar;
int GID_Jz = 2 + _XY * npar + _Z * __WIDTH * __HEIGHT * npar;
int GID_JBG = 3 + _XY * npar + _Z * __WIDTH * __HEIGHT * npar;
int GID_JNP = 4 + _XY * npar + _Z * __WIDTH * __HEIGHT * npar;

float fac_jac_loc = 1.f;
if ((I0 > 1e-5) & (M_loc >= 0))
    {fac_jac_loc = sqrt(M_loc)/abs(I0);}
else
    {fac_jac_loc = 0.f;}

float Jx_loc = - get_I(im_PSF,x,y,z,NP_loc,M_modif_loc,3,os) * os;
float Jy_loc = - get_I(im_PSF,x,y,z,NP_loc,M_modif_loc,2,os) * os;
float Jz_loc = - get_I(im_PSF,x,y,z,NP_loc,M_modif_loc,1,os);
float JBG_loc = 1.f;
float JNP_loc = (I0 - BG_loc) / NP_loc;

J[GID_Jx] = Jx_loc;
J[GID_Jy] = Jy_loc;
J[GID_Jz] = Jz_loc;
J[GID_JBG] = JBG_loc;
J[GID_JNP] = JNP_loc;

if (I0 > 0.f)
	{
	residual[GID_3D] = (I0 - M_loc) /I0;
	}
err_poi[GID_3D] = 2.f * get_error_poisson(I0,M_loc);
fac_jac[GID_3D] = fac_jac_loc;
}



extern "C" __global__ void test_indices(
float * J,
float * Jx,
float * Jy,
float * Jz,
float * JBG,
float * JNP,
int npar)
{
int _XY = _X + __WIDTH * _Y;
int GID_Jx = 0 + _XY * npar + _Z * __WIDTH * __HEIGHT * npar;
int GID_Jy = 1 + _XY * npar + _Z * __WIDTH * __HEIGHT * npar;
int GID_Jz = 2 + _XY * npar + _Z * __WIDTH * __HEIGHT * npar;
int GID_JBG = 3 + _XY * npar + _Z * __WIDTH * __HEIGHT * npar;
int GID_JNP = 4 + _XY * npar + _Z * __WIDTH * __HEIGHT * npar;
J[GID_Jx] = Jx[GID_3D];
J[GID_Jy] = Jy[GID_3D];
J[GID_Jz] = Jz[GID_3D];
J[GID_JBG] = JBG[GID_3D];
J[GID_JNP] = JNP[GID_3D];
}



