#include "headers/parser.h"

#include "headers/utils.h"

MLPData* deserialize(const std::string& data)
{
	const std::vector<std::string> lines = split(data, '\n');
	const std::vector<std::string> nplStr = split(lines[0], '_');
	std::vector<uint> npl;
	uint nplSize = nplStr.size();

	npl.reserve(nplSize);
	for (const std::string neuronStr : nplStr)
		npl.push_back((uint)stoi(neuronStr));

	const std::vector<std::string> weights = split(lines[1], '_');
	int weightIndex = 0;

	m3 W;
	m2 X;
	m2 deltas;

	W.reserve(nplSize);
	X.reserve(nplSize);
	deltas.reserve(nplSize);

	for (uint l = 0; l < nplSize; ++l)
	{
		if (l == 0)
		{
			W.push_back({});
			continue;
		}

		m2 v;

		v.reserve(npl[l - 1] + 1);

		for (uint i = 0; i < npl[l - 1] + 1; ++i)
		{
			m1 d;

			d.reserve(npl[l] + 1);

			for (uint j = 0; j < npl[l] + 1; ++j)
				d.push_back(stod(weights[weightIndex++]));

			v.push_back(d);
		}

		W.push_back(v);
	}

	for (uint l = 0; l < nplSize; ++l)
	{
		m1 x;
		m1 d;

		x.reserve(npl[l] + 1);
		d.reserve(npl[l] + 1);

		for (uint j = 0; j < npl[l] + 1; ++j)
		{
			x.push_back(j == 0 ? 1.0 : 0.0);
			d.push_back(0.0);
		}

		X.push_back(x);
		deltas.push_back(d);
	}

	return new MLPData(W, npl, X, deltas);
}

std::string serialize(const MLPData* model)
{
	std::string data = "";

	data += std::to_string(model->npl[0]);
	for (int i = 1; i < model->L + 1; ++i)
	{
		data += '_';
		data += std::to_string(model->npl[i]);
	}
	data += '\n';

	data += std::to_string(model->W[1][0][0]);
	for (int l = 1; l < model->W.size(); ++l)
	{
		for (int j = 0; j < model->W[l].size(); ++j)
		{
			for (int i = 0; i < model->W[l][j].size(); ++i)
			{
				if (l == 1 && j == 0 && i == 0) continue;

				data += '_';
				data += std::to_string(model->W[l][j][i]);
			}
		}
	}

	return data;
}