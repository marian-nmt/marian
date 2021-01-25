#include "marian.h"
#include "common/timer.h"

int main(int /*argc*/, char** /*argv*/) {
    using namespace marian;

    {
        auto g = New<ExpressionGraph>(true);
        g->setDevice({0, DeviceType::cpu});
#if 0 // this file is not a real test, just used for manual stuff. Disable here by hand for now.
        g->getBackend()->setInt16(false);
#endif
        g->reserveWorkspaceMB(2512);

        timer::AutoTimer timer;
        for(int i = 0; i < 100; ++i) {
            g->clear();

            auto x = g->constant({1, 4, 8, 256}, inits::glorotUniform());

            auto W1 = g->param("W1", {256, 2048}, inits::glorotUniform());
            auto b1 = g->param("b1", {1, 2048}, inits::glorotUniform());

            auto out = affine(x, W1, b1);

            for(int i = 2; i < 20; ++i) {
                auto Wi = g->param("W" + std::to_string(i), {2048, 2048}, inits::glorotUniform());
                auto bi = g->param("b" + std::to_string(i), {1, 2048}, inits::glorotUniform());

                out = relu(affine(out, Wi, bi));
            }

            auto Wn = g->param("Wn", {2048, 256}, inits::glorotUniform());
            auto bn = g->param("bn", {1, 256}, inits::glorotUniform());

            auto y = affine(out, Wn, bn);

            g->forward();
        }
    }

    {
        auto g = New<ExpressionGraph>(true);
        g->setDevice({0, DeviceType::cpu});
#if 0
        g->getBackend()->setInt16(true);
#endif
        g->reserveWorkspaceMB(2512);

        timer::AutoTimer timer;
        for(int i = 0; i < 100; ++i) {
            g->clear();

            auto x = g->constant({1, 4, 8, 256}, inits::glorotUniform());

            auto W1 = g->param("W1", {256, 2048}, inits::glorotUniform());
            auto b1 = g->param("b1", {1, 2048}, inits::glorotUniform());

            auto out = affine(x, W1, b1);

            for(int i = 2; i < 20; ++i) {
                auto Wi = g->param("W" + std::to_string(i), {2048, 2048}, inits::glorotUniform());
                auto bi = g->param("b" + std::to_string(i), {1, 2048}, inits::glorotUniform());

                out = relu(affine(out, Wi, bi));
            }

            auto Wn = g->param("Wn", {2048, 256}, inits::glorotUniform());
            auto bn = g->param("bn", {1, 256}, inits::glorotUniform());

            auto y = affine(out, Wn, bn);

            g->forward();
        }
    }

    {
        auto g = New<ExpressionGraph>(true);
        g->setDevice({0, DeviceType::cpu});
#if 0
        g->getBackend()->setInt8(true);
#endif
        g->reserveWorkspaceMB(2512);

        timer::AutoTimer timer;
        for(int i = 0; i < 100; ++i) {
            g->clear();

            auto x = g->constant({1, 4, 8, 256}, inits::glorotUniform());

            auto W1 = g->param("W1", {256, 2048}, inits::glorotUniform());
            auto b1 = g->param("b1", {1, 2048}, inits::glorotUniform());

            auto out = affine(x, W1, b1);

            for(int i = 2; i < 20; ++i) {
                auto Wi = g->param("W" + std::to_string(i), {2048, 2048}, inits::glorotUniform());
                auto bi = g->param("b" + std::to_string(i), {1, 2048}, inits::glorotUniform());

                out = relu(affine(out, Wi, bi));
            }

            auto Wn = g->param("Wn", {2048, 256}, inits::glorotUniform());
            auto bn = g->param("bn", {1, 256}, inits::glorotUniform());

            auto y = affine(out, Wn, bn);

            g->forward();
        }
    }

    return 0;
}
