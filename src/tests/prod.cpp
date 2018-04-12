#include "marian.h"


#include <boost/timer/timer.hpp>

int main(int argc, char** argv) {
    using namespace marian;

    {
        auto g = New<ExpressionGraph>(true, false);
        g->setDevice({0, DeviceType::cpu});
        g->reserveWorkspaceMB(2512);

        boost::timer::auto_cpu_timer timer;
        for(int i = 0; i < 100; ++i) {
            g->clear();

            auto x = g->constant({1, 4, 8, 256}, inits::glorot_uniform);

            auto W1 = g->param("W1", {256, 2048}, inits::glorot_uniform);
            auto b1 = g->param("b1", {1, 2048}, inits::glorot_uniform);

            auto out = affine(x, W1, b1);

            for(int i = 2; i < 20; ++i) {
                auto Wi = g->param("W" + std::to_string(i), {2048, 2048}, inits::glorot_uniform);
                auto bi = g->param("b" + std::to_string(i), {1, 2048}, inits::glorot_uniform);

                out = relu(affine(out, Wi, bi));
            }

            auto Wn = g->param("Wn", {2048, 256}, inits::glorot_uniform);
            auto bn = g->param("bn", {1, 256}, inits::glorot_uniform);

            auto y = affine(out, Wn, bn);

            g->forward();
        }
    }

    {
        auto g = New<ExpressionGraph>(true, true);
        g->setDevice({0, DeviceType::cpu});
        g->reserveWorkspaceMB(2512);

        boost::timer::auto_cpu_timer timer;
        for(int i = 0; i < 100; ++i) {
            g->clear();

            auto x = g->constant({1, 4, 8, 256}, inits::glorot_uniform);

            auto W1 = g->param("W1", {256, 2048}, inits::glorot_uniform);
            auto b1 = g->param("b1", {1, 2048}, inits::glorot_uniform);

            auto out = affine(x, W1, b1);

            for(int i = 2; i < 20; ++i) {
                auto Wi = g->param("W" + std::to_string(i), {2048, 2048}, inits::glorot_uniform);
                auto bi = g->param("b" + std::to_string(i), {1, 2048}, inits::glorot_uniform);

                out = relu(affine(out, Wi, bi));
            }

            auto Wn = g->param("Wn", {2048, 256}, inits::glorot_uniform);
            auto bn = g->param("bn", {1, 256}, inits::glorot_uniform);

            auto y = affine(out, Wn, bn);

            g->forward();
        }
    }


    return 0;
}
