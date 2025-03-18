#include <iostream>
#include <cstdint>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <map>
#include <valarray>
#include "nanoflann.h"

#include "tokenize.h"

class file_error: public std::exception
{
	std::string whatStr;
public:
	file_error(const std::string& whatIn): whatStr(whatIn)
	{}
	virtual const char* what() const noexcept override
	{
		return whatStr.c_str();
	}
};

struct LatentPoint
{
	float x;
	float y;
	float z;
	uint32_t classId;
	uint8_t r;
	uint8_t g;
	uint8_t b;
	uint8_t marked;
};

struct Box
{
	float x;
	float y;
	float z;
	float width;
	float height;
	float depth;
	float volume;
};

class LatentDataset
{
private:
	std::vector<LatentPoint> points;

public:
	LatentDataset() = default;
	LatentDataset(const std::filesystem::path& path);
	void load(const std::filesystem::path& path);
	void save(const std::filesystem::path& path);

	inline float kdtree_get_pt(const size_t index, int dim) const
	{
		switch(dim)
		{
			case 0:
				return points[index].x;
			case 1:
				return points[index].y;
			case 2:
				return points[index].z;
			default:
				assert(false);
		}
	}

	inline LatentPoint& operator[](size_t i) {return points[i];}
	inline const LatentPoint operator[](size_t i) const {return points[i];}
	inline const float* getPoint(size_t i) const {return reinterpret_cast<const float*>(&points[i]);}

	inline size_t kdtree_get_point_count() const { return points.size();}

	template <class BBOX>
	bool kdtree_get_bbox(BBOX&) const { return false;}

	std::vector<LatentDataset> splitByClass();

	void add(const LatentPoint& in) {points.push_back(in);}

	Box boundingBox() const;

	void remove(size_t index);
	void removeMarked();
};

LatentDataset::LatentDataset(const std::filesystem::path& path)
{
	load(path);
}

void LatentDataset::load(const std::filesystem::path& path)
{
	points.clear();

	std::fstream file(path, std::ios_base::in);
	if(!file.is_open())
		throw file_error("Unable to open " + path.string());

	size_t linenum = 1;
	std::string line;
	std::getline(file, line);

	while(file.good())
	{
		++linenum;
		std::getline(file, line);

		std::vector<std::string> tokens = tokenize(line, ',');
		if(tokens.size() == 1)
			continue;
		if(tokens.size() != 7)
			throw file_error("Input file invalid at line " + std::to_string(linenum));

		LatentPoint point;
		point.x = std::stof(tokens[0]);
		point.y = std::stof(tokens[1]);
		point.z = std::stof(tokens[2]);
		point.classId = std::stod(tokens[3]);
		point.r = std::clamp(0, 255, static_cast<int>(std::stod(tokens[3])));
		points.push_back(point);
	}
}

void LatentDataset::save(const std::filesystem::path& path)
{
	std::fstream file(path, std::ios_base::out);
	if(!file.is_open())
		throw file_error("Unable to open " + path.string());

	file<<std::scientific;
	file<<"x, y, z, class, r, g, b\n";
	for(const LatentPoint& point : points)
		file<<point.x<<", "<<point.y<<", "<<point.z<<", "<<point.classId<<", "<<point.r<<", "<<point.g<<", "<<point.b<<'\n';
}

void LatentDataset::remove(size_t index)
{
	points.erase(points.begin()+index);
}

void LatentDataset::removeMarked()
{
	for(size_t i = 0; i < kdtree_get_point_count(); ++i)
	{
		if(points[i].marked)
		{
			remove(i);
			--i;
		}
	}
}

std::vector<LatentDataset> LatentDataset::splitByClass()
{
	std::vector<LatentDataset> datasets;

	for(const LatentPoint& point : points)
	{
		if(point.classId+1 > datasets.size())
			datasets.resize(point.classId+1);
		datasets[point.classId].add(point);
	}

	return datasets;
}

Box LatentDataset::boundingBox() const
{
	std::valarray<float> xes(points.size());
	std::valarray<float> yes(points.size());
	std::valarray<float> zes(points.size());
	for(size_t i = 0; i < points.size(); ++i)
	{
		xes[i] = points[i].x;
		yes[i] = points[i].y;
		zes[i] = points[i].z;
	}

	Box box;
	box.x = xes.min();
	box.y = yes.min();
	box.z = zes.min();
	box.width = xes.max() - box.x;
	box.depth = yes.max() - box.y;
	box.height = zes.max() - box.z;
	box.volume = box.width*box.depth*box.height;
	return box;
}

float weightFn(float distance, float searchRadius)
{
	return 1.0f - (distance/searchRadius);
}

void pruneDataset(LatentDataset& dataset, float pruneRadius = 0.1)
{
	nanoflann::KDTreeSingleIndexAdaptorParams params;
	nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, LatentDataset>, LatentDataset, 3, size_t> kdtree(3, dataset, params);
	kdtree.buildIndex();

	size_t remvoved = 0;

	for(size_t i = 0; i < dataset.kdtree_get_point_count(); ++i)
	{
		if(dataset[i].marked)
			continue;

		std::vector<nanoflann::ResultItem<size_t, float>> results;
		kdtree.radiusSearch(dataset.getPoint(i), pruneRadius, results);

		for(auto result : results)
		{
			if(result.first == i)
				continue;
			if(dataset[result.first].classId == dataset[i].classId && !dataset[result.first].marked)
			{
				++remvoved;
				dataset[result.first].marked = true;
			}
		}
	}

	std::cout<<"pruned "<<remvoved<<" of "<<dataset.kdtree_get_point_count()<<'\n';

	dataset.removeMarked();
}

int main(int argc, char** argv)
{
	if(argc < 3)
	{
		std::cout<<"Usage: "<<argv[0]<<" [TRAIN_FILE] [TEST_FILE]\n";
		return 1;
	}

	LatentDataset trainDataset(argv[1]);
	LatentDataset testDataset(argv[2]);

	pruneDataset(trainDataset, 2);
	std::vector<LatentDataset> classDatasets = trainDataset.splitByClass();

	nanoflann::KDTreeSingleIndexAdaptorParams params;

	std::vector<nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, LatentDataset>, LatentDataset, 3, size_t>*> kdtrees;
	std::vector<float> distances;
	for(const LatentDataset& dataset : classDatasets)
	{
		kdtrees.push_back(new nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, LatentDataset>, LatentDataset, 3, size_t>(3, dataset, params));
		float density = dataset.boundingBox().volume / dataset.kdtree_get_point_count();
		float searchRadius = std::max(6.0f*std::pow(density, 1/3.0f), 2.0f);
		std::cout<<"searchRadius: "<<searchRadius<<'\n';
		distances.push_back(searchRadius);
		kdtrees.back()->buildIndex();
	}

	for(size_t i = 0; i < kdtrees.size(); ++i)
	{
		std::string name = "kdtree_" + std::to_string(i) + ".bin";
		std::fstream file("kdtree.bin", std::ios_base::out);
		if(!file.is_open())
		{
			std::cout<<"Can not open kdtree.bin for writeing";
			return 1;
		}
		kdtrees[i]->saveIndex(file);
		file.close();
	}

	size_t hit = 0;
	size_t miss = 0;
	size_t unkown = 0;

	for(size_t i = 0; i < testDataset.kdtree_get_point_count(); ++i)
	{
		std::cout<<"\ntesting point: "<<testDataset[i].x<<' '<<testDataset[i].y<<' '<<testDataset[i].z<<" class: "<<testDataset[i].classId<<'\n';
		std::vector<std::vector<nanoflann::ResultItem<size_t, float>>> results;
		size_t matches = 0;
		for(size_t classNum = 0; classNum < classDatasets.size(); ++classNum)
		{
			std::vector<nanoflann::ResultItem<size_t, float>> classResults;
			kdtrees[classNum]->radiusSearch(testDataset.getPoint(i), distances[classNum], classResults);
			results.push_back(classResults);
			matches += classResults.size();
		}

		if(results.empty())
		{
			++unkown;
			continue;
		}

		std::map<size_t, float> classesMap;
		for(size_t classNum = 0; classNum < results.size(); ++classNum)
		{
			for(nanoflann::ResultItem<size_t, float>& match : results[classNum])
			{
				auto search = classesMap.find(classDatasets[classNum][match.first].classId);
				if(search != classesMap.end())
					search->second += weightFn(match.second, distances[classNum]);
				else
					classesMap[classDatasets[classNum][match.first].classId] = weightFn(match.second, distances[classNum]);
			}
		}

		std::vector<std::pair<size_t, float>> classes(classesMap.begin(), classesMap.end());
		std::sort(classes.begin(), classes.end(), [](auto& a, auto& b){return a.second > b.second;});

		bool found = false;
		for(size_t j = 0; j < classes.size(); ++j)
		{
			std::cout<<"class: "<<classes[j].first<<" found: "<<classes[j].second<<'\n';
			if(classes[j].first == testDataset[i].classId && j <= 3)
				found = true;
		}

		hit += found;
		miss += !found;
	}

	std::cout<<"Acc: "<<hit/static_cast<double>(hit+miss)<<'\n';

	return 0;
}
