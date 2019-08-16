#include"models.h"
#include<random>

// Global variables
vector<Models::FollPar> C, G, S; // Mnemonics for coal-like, gas-like and solar-like followers.

vector<double> lincos = {150, 200, 220, 250, 275};
vector<double> quadcos = {0, 0.1, 0.2, 0.3, 0.5};
vector<double> emmcos = {25, 50, 100, 200, 300, 500};
vector<double> taxcaps = {0, 50, 100, 150, 200, 250};
vector<double> capacities = {50, 100, 130, 170, 200, 1000};

vector<double> demand_a = {275, 300, 325, 350};
vector<double> demand_b = {0.5, 0.6, 0.7, 0.9};

vector<Models::LeadAllPar> LeadersVec;

std::default_random_engine give;
std::uniform_int_distribution<int> binaryRandom(0, 1);
std::uniform_int_distribution<int> intRandom(0, 1e5);

Models::FollPar operator+(Models::FollPar a, Models::FollPar b) {
    vector<double> costs_quad = a.costs_quad;
    vector<double> costs_lin = a.costs_lin;
    vector<double> capacities = a.capacities;
    vector<double> emission_costs = a.emission_costs;
    vector<double> tax_caps = a.tax_caps;
    vector<string> names = a.names;

    costs_quad.insert(costs_quad.end(), b.costs_quad.begin(), b.costs_quad.end());
    costs_lin.insert(costs_lin.end(), b.costs_lin.begin(), b.costs_lin.end());
    capacities.insert(capacities.end(), b.capacities.begin(), b.capacities.end());
    emission_costs.insert(emission_costs.end(), b.emission_costs.begin(), b.emission_costs.end());
    tax_caps.insert(tax_caps.end(), b.tax_caps.begin(), b.tax_caps.end());
    names.insert(names.end(), b.names.begin(), b.names.end());

    return Models::FollPar(costs_quad, costs_lin, capacities, emission_costs, tax_caps, names);
}


Models::FollPar makeFollPar(int costParam = 0, int polluting = 0, int capac = 0) {
    costParam = costParam % lincos.size();
    polluting = polluting % 3;
    capac = capac % capacities.size();;

    Models::FollPar FP_temp;
    FP_temp.costs_lin = {lincos[costParam]};
    FP_temp.costs_quad = {quadcos[quadcos.size() - costParam - 1]};

    FP_temp.capacities = {capacities[capac]};

    int r;

    switch (polluting) {
        case 0:                // Solar type
            r = binaryRandom(give);
            FP_temp.emission_costs = {emmcos[r]};
            r = binaryRandom(give);
            FP_temp.tax_caps = {emmcos[r]};
            FP_temp.names = {"S" + to_string(S.size())};
            S.push_back(FP_temp);
            break;
        case 1:                    // Natural Gas type
            r = binaryRandom(give);
            FP_temp.emission_costs = {emmcos[2 + r]};
            r = binaryRandom(give);
            FP_temp.tax_caps = {emmcos[2 + r]};
            FP_temp.names = {"G" + to_string(G.size())};
            G.push_back(FP_temp);
            break;
        case 2:                // Coal type
            r = binaryRandom(give);
            FP_temp.emission_costs = {emmcos[4 + r]};
            r = binaryRandom(give);
            FP_temp.tax_caps = {emmcos[4 + r]};
            FP_temp.names = {"C" + to_string(C.size())};
            C.push_back(FP_temp);
            break;
    };
    return FP_temp;
}

Models::LeadAllPar makeLeader(bool cc, bool gg, bool ss, int cC = 5, int gC = 5, int sC = 5, double price_lim = 0.9) {
    static int country = 0;
    bool madeFirst{false};
    Models::FollPar F;
    int costParam = intRandom(give) % (lincos.size() - 2);
    if (cc) {
        F = makeFollPar(costParam, 2, cC);
        madeFirst = true;
    }
    if (gg) {
        costParam++;
        Models::FollPar Ftemp;
        (madeFirst ? Ftemp : F) = makeFollPar(costParam, 1, gC);
        if (madeFirst)
            F = F + Ftemp;
        madeFirst = true;
    }
    if (ss) {
        costParam += 2;
        Models::FollPar Ftemp;
        (madeFirst ? Ftemp : F) = makeFollPar(costParam, 0, sC);
        if (madeFirst)
            F = F + Ftemp;
        madeFirst = true;
    }

    double a = demand_a[intRandom(give) % demand_a.size()];
    double b = demand_b[intRandom(give) % demand_b.size()];

    double total_cap{0}, price_limit;
    for (auto v:F.capacities)
        if (v == capacities.back())
            total_cap += capacities[4];
        else
            total_cap += v;
    /*
    price_limit = (a - b*total_cap)/price_lim;

    price_limit = price_limit > a*1.05? a * price_lim:price_limit;
    price_limit = (price_limit < 0)? a*price_lim:price_limit;
    */
    price_limit = price_lim * a;

    Models::LeadAllPar Country(F.capacities.size(), "Country_" + to_string(country++), F, {a, b},
                               {-1, -1, price_limit});
    LeadersVec.push_back(Country);

    return Country;

}


void MakeCountry() {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                if (i == 0 && j == 0 && k == 0)
                    break;
                for (double perclim = 0.8; perclim <= 0.95; perclim += 0.05) {
                    makeLeader(i, j, k, intRandom(give), intRandom(give), intRandom(give), perclim);
                    makeLeader(i, j, k, intRandom(give), intRandom(give), intRandom(give), perclim);
                    makeLeader(i, j, k, intRandom(give), intRandom(give), intRandom(give), perclim);
                    makeLeader(i, j, k, intRandom(give), intRandom(give), intRandom(give), perclim);
                }

            }
        }
    }
}

void MakeInstance(int nCountries = 2) {
    static int count{0};
    MakeCountry();
    int nNet = LeadersVec.size();
    vector<Models::LeadAllPar> cVec;
    cout << "Instance " << count << " with ";
    for (int i = 0; i < nCountries; i++) {
        auto val = intRandom(give) % nNet;
        cVec.push_back(
                LeadersVec[val]
        );
        cout << val << "\t";
    }
    cout << endl;
    arma::sp_mat TrCo(nCountries, nCountries);
    for (int i = 0; i < nCountries; i++)
        for (int j = 0; j < nCountries; j++)
            if (i != j)
                TrCo(i, j) = 1;
            else
                TrCo(i, j) = 0;

    Models::EPECInstance Inst(cVec, TrCo);
    Inst.save("dat/Instance_" + to_string(count++));
}

int main() {
    for (int i = 0; i < 50; ++i) MakeInstance(4);
    for (int i = 0; i < 50; ++i) MakeInstance(5);
    return 0;
}
