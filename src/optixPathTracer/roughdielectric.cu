/*
Rough dielectric materials
https://dl.acm.org/citation.cfm?id=2383874 [Walter et al.]
*/

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu_matrix_namespace.h>
#include "helpers.h"
#include "prd.h"
#include "random.h"
#include "rt_function.h"
#include "material_parameters.h"
#include "state.h"

using namespace optix;

/* HELPERS */

/* Return 1 if the input >= 0 and -1 otherwise. */
RT_FUNCTION inline float sgn(float x)
{
	return (x >= 0.0f) ? 1.0f : -1.0f;
}

/* Convert the user-specified roughness to the roughness value suitable
   for each model. This is done in a way such that even different models
   produce similar appearances for the same user-specified roughness. */
RT_FUNCTION float alphaConversion(float roughness, DistType dist)
{
	const float minAlpha = 1e-3f;
	const float b2g = 1.1312f; // Walter et al. (Fig. 12)
	float ggx_alpha = fmax(roughness, minAlpha);
	float beck_alpha = ggx_alpha / b2g;

	switch (dist)
	{
	case DistType::Beckmann:
		return beck_alpha;
	case DistType::GGX:
		return ggx_alpha;
	case DistType::Phong:
		return 2.0f / (beck_alpha * beck_alpha) - 2.0f;
	default:
		return ggx_alpha;
	}
}

/* Sample the polar angle and the azimuthal angle.
   Then convert them into the 3D Euclidean coordinates (local frame). */
RT_FUNCTION float3 sample(float r1, float r2, float alpha, DistType dist)
{
	float cosTheta = 0.0f; // cosine of the polar angle
	const float phi = r1 * 2.0f * M_PIf; // azimuthal angle
	const float alphaSqr = alpha * alpha;

	switch (dist)
	{
	case DistType::Beckmann:
	{
		cosTheta = 1.0f / sqrtf(fmax(1.0f - alphaSqr * logf(1.0f - r2), 0.0f));
		break;
	}
	case DistType::GGX:
	{
		cosTheta = 1.0f / sqrtf(fmax(1.0f + alphaSqr * r2 / (1.0f - r2), 0.0f));
		break;
	}
	case DistType::Phong:
	{
		cosTheta = pow(r2, 1.0f / (alpha + 2.0f));
		break;
	}
	}

	const float r = sqrtf(fmax(1.0f - cosTheta * cosTheta, 0.0f));
	return optix::make_float3(cosf(phi) * r, sinf(phi) * r, cosTheta);
}

/* Fresnel (reflection) coefficient.
   This term describes how much of an electromagnetic wave is reflected
   by an impedance discontinuity in the transmission medium
   [Wikipedia: Reflection coefficient]. */
RT_FUNCTION float fresnelTerm(float cosThetaI, float invEta, float &cosThetaT)
{
	if (cosThetaI < 0.0f)
		cosThetaI = -cosThetaI;
	const float cosThetaTSqr = 1.0f - invEta * invEta * (1.0f - cosThetaI * cosThetaI);
	if (cosThetaTSqr < 0.0f) // total reflection
	{
		cosThetaT = 0.0f; // meaningless
		return 1.0f;
	}

	cosThetaT = sqrtf(fmax(cosThetaTSqr, 0.0f)); // = abs(cosThetaT), to be precise
	const float Rs = (cosThetaT - invEta * cosThetaI) / (cosThetaT + invEta * cosThetaI);
	const float Rp = (invEta * cosThetaT - cosThetaI) / (invEta * cosThetaT + cosThetaI);
	return 0.5f * (Rs * Rs + Rp * Rp);
}

/* Wrapper for the Fresnel coefficient. */
RT_FUNCTION float fresnelTerm(float cosThetaI, float invEta)
{
	float cosThetaT;
	return fresnelTerm(cosThetaI, invEta, cosThetaT);
}

/* Microfacet distribution function. */
RT_FUNCTION float D(float cosThetaM, float alpha, DistType dist)
{
	if (cosThetaM <= 0.0f)
		return 0.0f;

	const float alphaSqr = alpha * alpha;
	const float cosThetaSqr = cosThetaM * cosThetaM;
	const float cosThetaQd = cosThetaSqr * cosThetaSqr;
	const float beckmannExp = -(1.0f / cosThetaSqr - 1.0f) / alphaSqr;
	const float ggxDivisor = (1.0f - beckmannExp);

	switch (dist)
	{
	case DistType::Beckmann:
		return optix::expf(beckmannExp) * M_1_PIf / (alphaSqr * cosThetaQd);
	case DistType::GGX:
		return M_1_PIf / (cosThetaQd * ggxDivisor * ggxDivisor);
	case DistType::Phong:
		return 0.5f * (alpha + 2) * M_1_PIf * powf(cosThetaM, alpha);
	default:
		return 0.0f;
	}
}

/* Unidirectional shadow-masking function. */
RT_FUNCTION float G1(float3 v, float3 m, float3 n, float alpha, DistType dist)
{
	const float cosTheta = dot(v, n);
	if (dot(v, m) / cosTheta <= 0.0f)
		return 0.0f;

	const float tanTheta = abs(sqrtf(fmax(1 - cosTheta * cosTheta, 0.0f)) / cosTheta); // = abs(invTanTheta), to be precise
	const float alphaTan = alpha * tanTheta;
	float a;
	if (dist == DistType::Beckmann)
		a = 1.0f / alphaTan;
	else if (dist == DistType::Phong)
		a = sqrtf(1 + 0.5f * alpha) / tanTheta;

	switch (dist)
	{
	case DistType::Beckmann:
		if (a < 1.6f)
			return (3.535f * a + 2.181f * a * a) / (1.0f + 2.276f * a + 2.577f * a * a);
		else
			return 1.0f;
	case DistType::GGX:
		return 2.0f / (1.0f + sqrtf(1.0f + alphaTan * alphaTan));
	case DistType::Phong:
		if (a < 1.6f)
			return (3.535f * a + 2.181f * a * a) / (1.0f + 2.276f * a + 2.577f * a * a);
		else
			return 1.0f;
	default:
		return 0.0f;
	}
}

/* Bidirectional shadow-masking function. */
RT_FUNCTION float G(float3 i, float3 o, float3 m, float3 n, float alpha, DistType dist)
{
	const float g1 = G1(i, m, n, alpha, dist);
	if (g1 == 0.0f)
		return 0.0f;

	const float g2 = G1(o, m, n, alpha, dist);
	if (g2 == 0.0f)
		return 0.0f;

	return abs(g1 * g2); // take abs to prevent unintentional numerical errors
}


/* MAIN ROUTINES */

/* Sample an transmitted or an reflected directional vector. */
RT_CALLABLE_PROGRAM void Sample(MaterialParameter &mat, State &state, PerRayData_radiance &prd)
{
	/* World frame vectors */
	const float3 N = state.normal; // shading normal, pointing to the outside
	const float3 V = prd.wo; // = -ray.direction, pointing away from the intersection point

	/* Roughness scaling and conversion */
	const float VDotN = optix::dot(V, N);
	float sampleAlphaScale = 1.2f - 0.2f * sqrtf(abs(VDotN)); // roughness scaling trick by Walter et al. (Chap. 5.3, p.8)
	float sampleAlpha = alphaConversion(sampleAlphaScale * mat.roughness, mat.dist);

	/* Microfacet-normal sampling */
	const float r1 = rnd(prd.seed);
	const float r2 = rnd(prd.seed);
	float3 m = sample(r1, r2, sampleAlpha, mat.dist); // local frame
	optix::Onb onb(N);
	onb.inverse_transform(m); // covnert a local frame to the world frame

	/* Fresnel term computing */
	const float VDotM = optix::dot(V, m);
	float cosThetaT = 0.0f; // transmission angle ([0, pi/2])
	const float invEta = VDotM > 0.0f ? mat.extIOR / mat.intIOR : mat.intIOR / mat.extIOR;
	const float F = fresnelTerm(VDotM, invEta, cosThetaT);

	const float p = rnd(prd.seed);
	if (p <= F)
	{
		prd.origin = state.fhp;
		prd.bsdfDir = 2.0f * VDotM * m - V;

		// update path feature
		prd.roughness = alphaConversion(mat.roughness, mat.dist);
		prd.bounce_type = BSDF_REFLECTION | BSDF_GLOSSY;

		/* Sanity check */
		if (dot(V, N) * dot(prd.bsdfDir, N) <= 0.0f) // should be reflected, but it wasn't
			prd.done = true;
	}
	else
	{
		prd.origin = state.bhp;
		prd.bsdfDir = (invEta * VDotM - sgn(VDotM) * cosThetaT) * m - invEta * V;

		// update path feature
		prd.roughness = alphaConversion(mat.roughness, mat.dist);
		prd.bounce_type = BSDF_TRANSMISSION | BSDF_GLOSSY;

		/* Sanity check */
		if (dot(V, N) * dot(prd.bsdfDir, N) >= 0.0f) // should be refracted, but it wasn't
			prd.done = true;
	}
}

/* Evaluate pdf (sampled direction). */
RT_CALLABLE_PROGRAM void Pdf(MaterialParameter &mat, State &state, PerRayData_radiance &prd)
{
	/* Sanity check */
	if (prd.done)
	{
		prd.pdf = 0.0f;
		return;
	}

	/* World frame vectors */
	const float3 N = state.normal;
	const float3 i = prd.wo;
	const float3 o = prd.bsdfDir;

	/* Roughness scaling and conversion */
	const float iDotN = dot(i, N);
	float sampleAlphaScale = 1.2f - 0.2f * sqrtf(abs(iDotN)); // roughness scaling trick by Walter et al. (Chap. 5.3, p.8)
	float sampleAlpha = alphaConversion(sampleAlphaScale * mat.roughness, mat.dist);

	if (iDotN * dot(o, N) > 0) // reflection
	{
		/* Half vector */
		const float3 m = normalize(i + o) * sgn(iDotN);

		/* Fresnel term computing */
		const float iDotm = dot(i, m);
		const float invEta = iDotm > 0.0f ? mat.extIOR / mat.intIOR : mat.intIOR / mat.extIOR;
		const float F = fresnelTerm(iDotm, invEta);

		/* Microfacet distribution evaluating */
		const float mDotN = dot(m, N);
		const float microPdf = D(mDotN, sampleAlpha, mat.dist);

		/* Macrosurface distribution evaluating */
		prd.pdf = abs(F * microPdf * mDotN / (4 * iDotm));
	}
	else // refraction
	{
		/* Half vector */
		const float eta = iDotN > 0.0f ? mat.intIOR / mat.extIOR : mat.extIOR / mat.intIOR;
		const float3 m = -normalize(i + eta * o);

		/* Fresnel term computing */
		const float iDotm = dot(i, m);
		const float invEta = iDotm > 0.0f ? mat.extIOR / mat.intIOR : mat.intIOR / mat.extIOR;
		const float F = fresnelTerm(iDotm, invEta);

		/* Microfacet distribution evaluating */
		const float mDotN = dot(m, N);
		const float microPdf = D(mDotN, sampleAlpha, mat.dist);

		/* Macrosurface distribution evaluating */
		const float oDotm = dot(o, m);
		const float divisor = iDotm / eta + oDotm;
		prd.pdf = abs((1 - F) * microPdf * mDotN * oDotm / (divisor * divisor));
	}
}

/* Evaluate f_s(i,o,n)*|o*n|. */
RT_CALLABLE_PROGRAM float3 Eval(MaterialParameter &mat, State &state, PerRayData_radiance &prd)
{
	/* Sanity check */
	if (prd.done)
	{
		return make_float3(0.0f);
	}

	/* World frame vectors */
	const float3 N = state.normal;
	const float3 i = prd.wo;
	const float3 o = prd.bsdfDir;

	/* Roughness scaling and conversion */
	const float iDotN = dot(i, N);
	float alpha = alphaConversion(mat.roughness, mat.dist);

	if (iDotN * dot(o, N) > 0) // reflection
	{
		/* Half vector */
		const float3 m = normalize(i + o) * sgn(iDotN);

		/* Fresnel term computing */
		const float iDotm = dot(i, m);
		const float invEta = iDotm > 0.0f ? mat.extIOR / mat.intIOR : mat.intIOR / mat.extIOR;
		const float F = fresnelTerm(iDotm, invEta);

		/* Microfacet distribution evaluating */
		const float mDotN = dot(m, N);
		const float microPdf = D(mDotN, alpha, mat.dist);

		/* Bidirectional shadow-masking function */
		const float Geo = G(i, o, m, N, alpha, mat.dist);

		/* BSDF*cosine evaluating */
		const float f = abs(F * Geo * microPdf / (4 * iDotN));

		// update path feature
		prd.thpt_at_vtx = make_float3(f);

		return prd.thpt_at_vtx;
	}
	else // refraction
	{
		/* Half vector */
		const float eta = iDotN > 0.0f ? mat.intIOR / mat.extIOR : mat.extIOR / mat.intIOR;
		const float3 m = -normalize(i + eta * o);

		/* Fresnel term computing */
		const float iDotm = dot(i, m);
		const float invEta = iDotm > 0.0f ? mat.extIOR / mat.intIOR : mat.intIOR / mat.extIOR;
		const float F = fresnelTerm(iDotm, invEta);

		/* Microfacet distribution evaluating */
		const float mDotN = dot(m, N);
		const float microPdf = D(mDotN, alpha, mat.dist);

		/* Bi-directional shadow-masking function */
		const float Geo = G(i, o, m, N, alpha, mat.dist);

		/* BSDF*cosine evaluating */
		const float oDotm = dot(o, m);
		const float divisor = iDotm / eta + oDotm;
		const float f = abs((1 - F) * Geo * microPdf * iDotm * oDotm / (divisor * divisor) / iDotN);

		// update path feature
		prd.thpt_at_vtx = mat.color * make_float3(f);

		return prd.thpt_at_vtx;
	}
}


RT_CALLABLE_PROGRAM float3 EvalDiffuse(MaterialParameter &mat, State &state, PerRayData_radiance &prd)
{
	return make_float3(0.0f);
}