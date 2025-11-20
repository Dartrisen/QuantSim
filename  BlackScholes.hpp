#pragma once

#include "DiffusionProcess.hpp"

typedef double Rate;

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
