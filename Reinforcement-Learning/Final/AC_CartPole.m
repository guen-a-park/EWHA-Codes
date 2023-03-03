%% Create Environment Interface

env = rlPredefinedEnv("CartPole-Discrete")

env.PenaltyForFalling = -15;
env.XThreshold = 2;

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

rng(0)

%% Create AC Agent

criticNetwork = [
    imageInputLayer([4 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(1,'Name','CriticFC')];

criticOpts = rlRepresentationOptions('LearnRate',1e-2,'GradientThreshold',1); %8e-3

critic = rlRepresentation(criticNetwork,obsInfo,'Observation',{'state'},criticOpts);

actorNetwork = [
    imageInputLayer([4 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(2,'Name','action')];

actorOpts = rlRepresentationOptions('LearnRate',1e-2,'GradientThreshold',1);  %8e-3

actor = rlRepresentation(actorNetwork,obsInfo,actInfo,...
    'Observation',{'state'},'Action',{'action'},actorOpts);

agentOpts = rlACAgentOptions(...
    'NumStepsToLookAhead',32, ...
    'DiscountFactor',0.99);

agent = rlACAgent(actor,critic,agentOpts);


%% Train Agent

trainOpts = rlTrainingOptions(...
    'MaxEpisodes',1000,...
    'MaxStepsPerEpisode',500,...
    'Verbose',true,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',500,...
    'ScoreAveragingWindowLength',15); 

plot(env)

doTraining = true;

if doTraining    
    % Train the agent.
    trainingStats = train(agent,env,trainOpts)
else
    % Load pretrained agent for the example.
    load('MATLABCartpoleAC.mat','agent');
end

%% Simulate AC Agent
plot(env)
simOptions = rlSimulationOptions('MaxSteps',1000);
experience = sim(env,agent,simOptions);
totalReward = sum(experience.Reward)

