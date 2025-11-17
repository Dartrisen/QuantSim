/***************************************************************************************************
General diffusion process class
This class describes a stochastic process governed by dx(t) = mu(t, x(t))dt + sigma(t, x(t))dz(t).
***************************************************************************************************/
#pragma once

#include <memory>
#include <cassert>

typedef double Time;
typedef double Rate;

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

class BlackScholesProcess : public DiffusionProcess {
public:
    BlackScholesProcess(Rate r, double sigma, double s0)
        : DiffusionProcess(std::log(s0)), r_(r), sigma_(sigma)
    {
        assert(s0 > 0.0 && "S0 must be positive");
        assert(sigma_ >= 0.0 && "sigma must be non-negative");
    }

    // dY = (r - 0.5*sigma^2) dt + sigma dW
    double drift(Time /*t*/, double /*y*/) const noexcept override {
        return r_ - 0.5 * sigma_ * sigma_;
    }
    double diffusion(Time /*t*/, double /*y*/) const noexcept override {
        return sigma_;
    }

    std::unique_ptr<DiffusionProcess> clone() const override {
        return std::make_unique<BlackScholesProcess>(*this);
    }

    // accessors
    double r() const noexcept { return r_; }
    double sigma() const noexcept { return sigma_; }

private:
    double r_, sigma_;
};
