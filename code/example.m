string = 'a';
string2 = 'b';
stringtot = [string int2str(1) string2];

dataset = load('..\datasets\dataset.mat'); 
dataset = table2array(dataset.dataset);
X = dataset(:,5:end)';
y_arousal = dataset(:,3)';
y_valence = dataset(:,4)';


% Create plots

for i=1:9
    figure
    t = tiledlayout(3,2); % Requires R2019b or later
    nexttile;
    plot(X(i,:), y_arousal, 'k+');
    nexttile
    plot(X(i+1,:), y_arousal, 'k+');
    nexttile;
    plot(X(i+2,:), y_arousal, 'k+');
    nexttile
    plot(X(i+3,:), y_arousal, 'k+');
    nexttile
    plot(X(i+4,:), y_arousal, 'k+');
    nexttile
    plot(X(i+5,:), y_arousal, 'k+');
end