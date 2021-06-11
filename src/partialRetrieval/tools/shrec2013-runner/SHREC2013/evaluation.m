load datasetSHREC2013_data;

distances = dlmread('results.res', ' ');
distances(:,361) = [];

QR = zeros(1,7200);

for i = 1 : 7200
    A = distances(i,:);
    [X,Y] = sort(A);
    pos = find(Y==query(i).indiceTarget);
    QR(i) = pos;
end

MQR = mean(QR);
WMQR= sum(QR.*ratio)/sum(ratio);

accumClass = zeros(1, 20);
accumClassWeighted = zeros(1, 20);
sumRatio = zeros(1,20);

for i = 1 : 7200
    A = distances(i,:);
    [X,Y] = sort(A);
    pos = find(Y==query(i).indiceTarget);
    accumClass(query(i).class) = accumClass(query(i).class) + pos;
    accumClassWeighted(query(i).class) = accumClassWeighted(query(i).class) + pos*ratio(i);
    sumRatio(query(i).class) = sumRatio(query(i).class) + ratio(i);
end


accumClass = accumClass/360;
accumClassWeighted = accumClassWeighted./sumRatio;
%fid = fopen('partiality.txt', 'w');
%fprintf(fid, '%d\n', length(ratio));

%for i  = 1 : length(ratio)
%    fprintf(fid, '%f\n', ratio(i));
%end

%fclose(fid);