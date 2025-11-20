/***************************************************************************************************
General diffusion process class
This class describes a stochastic process governed by dx(t) = mu(t, x(t))dt + sigma(t, x(t))dz(t).
***************************************************************************************************/
#pragma once

#include <cassert>
#include <memory>

typedef double Time;

class DiffusionProcess {
public:
    using ptr = std::unique_ptr<DiffusionProcess>;
    explicit DiffusionProcess(double x0) noexcept : x0_(x0) {}
    virtual ~DiffusionProcess() = default;

    double x0() const noexcept { return x0_; }

    virtual double drift(Time t, double x) const noexcept = 0;
    virtual double diffusion(Time t, double x) const noexcept = 0;

    virtual double expectation(Time t0, double x0, Time dt) const noexcept {
        return x0 + drift(t0, x0) * dt;
    }
    virtual double variance(Time t0, double x0, Time dt) const noexcept {
        double s = diffusion(t0, x0);
        return s * s * dt;
    }

    virtual ptr clone() const = 0;

private:
    double x0_;
};
